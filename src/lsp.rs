//! LSP Client for rust-analyzer semantic enrichment.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use lsp_types::{
    ClientCapabilities, DidChangeTextDocumentParams, DidOpenTextDocumentParams,
    DocumentSymbolParams, DocumentSymbolResponse, GotoDefinitionParams, GotoDefinitionResponse,
    HoverParams, Hover, InitializeParams, InitializeResult, Location, LocationLink,
    Position, Range, ReferenceParams, ReferenceContext, SymbolInformation,
    SymbolKind, TextDocumentIdentifier, TextDocumentItem, Uri, WorkspaceFolder,
    WorkDoneProgressParams,
};

pub use lsp_types::{
    CallHierarchyIncomingCall, CallHierarchyIncomingCallsParams, CallHierarchyItem,
    CallHierarchyOutgoingCall, CallHierarchyOutgoingCallsParams, CallHierarchyPrepareParams,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Child;
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{debug, error, info, warn};
use url::Url;

/// LSP message with JSON-RPC framing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspMessage {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<LspError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Pending request waiting for response
struct PendingRequest {
    sender: tokio::sync::oneshot::Sender<Result<Value>>,
}

/// Parameter type for textDocument/implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GotoImplementationParams {
    #[serde(flatten)]
    pub text_document_position_params: lsp_types::TextDocumentPositionParams,
    #[serde(flatten)]
    pub work_done_progress_params: WorkDoneProgressParams,
    #[serde(flatten)]
    pub partial_result_params: lsp_types::PartialResultParams,
}

/// LSP Client for communicating with rust-analyzer
pub struct LspClient {
    process: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<tokio::process::ChildStdin>>>,
    pending_requests: Arc<RwLock<HashMap<u64, PendingRequest>>>,
    request_id: Arc<std::sync::atomic::AtomicU64>,
    workspace_root: PathBuf,
    initialized: Arc<std::sync::atomic::AtomicBool>,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl LspClient {
    /// Create a new LSP client for the given workspace root
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            process: Arc::new(Mutex::new(None)),
            stdin: Arc::new(Mutex::new(None)),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            request_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            workspace_root,
            initialized: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            shutdown_tx: None,
        }
    }

    /// Convert a file path to a file:// URI
    fn path_to_uri(path: &Path) -> Result<Uri> {
        let url = Url::from_file_path(path)
            .map_err(|_| anyhow::anyhow!("Invalid file path"))?;
        let uri_str = url.to_string();
        Ok(Uri::from_str(&uri_str)?)
    }

    /// Start rust-analyzer and initialize the LSP connection
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting rust-analyzer for workspace: {:?}", self.workspace_root);

        // Check if rust-analyzer is available
        let ra_path = which::which("rust-analyzer")
            .context("rust-analyzer not found in PATH. Install it with: rustup component add rust-analyzer")?;

        let mut cmd = Command::new(ra_path);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(&self.workspace_root);

        let mut child = tokio::process::Command::from(cmd)
            .spawn()
            .context("Failed to spawn rust-analyzer")?;

        let stdin = child.stdin.take().context("Failed to get stdin")?;
        let stdout = child.stdout.take().context("Failed to get stdout")?;

        self.stdin = Arc::new(Mutex::new(Some(stdin)));
        self.process = Arc::new(Mutex::new(Some(child)));

        // Start response reader task
        let pending_requests = self.pending_requests.clone();
        
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut buffer = Vec::new();
            
            loop {
                // Read headers
                let mut content_length = 0;
                loop {
                    let mut line = String::new();
                    match reader.read_line(&mut line).await {
                        Ok(0) => return, // EOF
                        Ok(_) => {
                            if line.trim().is_empty() {
                                break; // End of headers
                            }
                            if line.starts_with("Content-Length:") {
                                content_length = line["Content-Length:".len()..].trim().parse().unwrap_or(0);
                            }
                        }
                        Err(e) => {
                            error!("Error reading LSP header: {}", e);
                            return;
                        }
                    }
                }
                
                if content_length == 0 {
                    continue;
                }
                
                // Read body
                buffer.resize(content_length, 0);
                if let Err(e) = reader.read_exact(&mut buffer).await {
                    error!("Error reading LSP body: {}", e);
                    continue;
                }
                
                let body_str = String::from_utf8_lossy(&buffer);
                debug!("LSP Response: {}", body_str);
                
                if let Ok(msg) = serde_json::from_str::<LspMessage>(&body_str) {
                    if let Some(id) = msg.id.as_ref().and_then(|v| v.as_u64()) {
                        let mut pending = pending_requests.write().await;
                        if let Some(pending_req) = pending.remove(&id) {
                            if let Some(error) = msg.error {
                                let _ = pending_req.sender.send(Err(anyhow::anyhow!("LSP error {}: {}", error.code, error.message)));
                            } else if let Some(result) = msg.result {
                                let _ = pending_req.sender.send(Ok(result));
                            } else {
                                let _ = pending_req.sender.send(Err(anyhow::anyhow!("No result or error in response")));
                            }
                        }
                    } else if msg.method.is_some() {
                        // Handle notifications (not implemented yet)
                        debug!("LSP Notification: {:?}", msg.method);
                    }
                }
            }
        });

        // Wait a bit for process to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Initialize
        self.initialize().await?;

        Ok(())
    }

    /// Get next request ID
    fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    /// Send a request and wait for response
    async fn send_request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id();
        let request = LspMessage {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(id.into())),
            method: Some(method.to_string()),
            params: Some(params),
            result: None,
            error: None,
        };

        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut pending = self.pending_requests.write().await;
            pending.insert(id, PendingRequest { sender: tx });
        }

        self.write_message(&request).await?;

        // Wait for response with timeout
        match tokio::time::timeout(Duration::from_secs(30), rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Request channel closed")),
            Err(_) => Err(anyhow::anyhow!("Request timeout")),
        }
    }

    /// Send a notification (no response expected)
    async fn send_notification(&self, method: &str, params: Value) -> Result<()> {
        let notification = LspMessage {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: Some(method.to_string()),
            params: Some(params),
            result: None,
            error: None,
        };
        self.write_message(&notification).await
    }

    /// Write a message to the LSP server
    async fn write_message(&self, msg: &LspMessage) -> Result<()> {
        let body = serde_json::to_string(msg)?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        let message = format!("{}{}", header, body);

        let mut stdin_guard = self.stdin.lock().await;
        if let Some(stdin) = stdin_guard.as_mut() {
            stdin.write_all(message.as_bytes()).await?;
            stdin.flush().await?;
        } else {
            return Err(anyhow::anyhow!("LSP stdin not available"));
        }
        Ok(())
    }

    /// Initialize the LSP connection
    async fn initialize(&mut self) -> Result<()> {
        let root_uri = Self::path_to_uri(&self.workspace_root)?;

        let init_params = InitializeParams {
            capabilities: ClientCapabilities::default(),
            workspace_folders: Some(vec![WorkspaceFolder {
                uri: root_uri.clone(),
                name: self.workspace_root
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("workspace")
                    .to_string(),
            }]),
            root_uri: Some(root_uri),
            ..Default::default()
        };

        let result: InitializeResult = serde_json::from_value(
            self.send_request("initialize", serde_json::to_value(init_params)?).await?
        )?;

        info!("LSP initialized: {:?}", result.capabilities);

        // Send initialized notification
        self.send_notification("initialized", json!({})).await?;

        self.initialized.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Check if LSP is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Open a text document in the LSP
    pub async fn did_open(&self, file_path: &Path, content: &str) -> Result<()> {
        let uri = Self::path_to_uri(file_path)?;

        let params = DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri,
                language_id: "rust".to_string(),
                version: 1,
                text: content.to_string(),
            },
        };

        self.send_notification("textDocument/didOpen", serde_json::to_value(params)?).await
    }

    /// Notify LSP of document changes
    pub async fn did_change(&self, file_path: &Path, content: &str, version: i32) -> Result<()> {
        let uri = Self::path_to_uri(file_path)?;

        let params = DidChangeTextDocumentParams {
            text_document: lsp_types::VersionedTextDocumentIdentifier {
                uri,
                version,
            },
            content_changes: vec![lsp_types::TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text: content.to_string(),
            }],
        };

        self.send_notification("textDocument/didChange", serde_json::to_value(params)?).await
    }

    /// Get document symbols for a file
    pub async fn document_symbols(&self, file_path: &Path) -> Result<Vec<DocumentSymbol>> {
        let uri = Self::path_to_uri(file_path)?;

        let params = DocumentSymbolParams {
            text_document: TextDocumentIdentifier { uri },
            work_done_progress_params: WorkDoneProgressParams::default(),
            partial_result_params: Default::default(),
        };

        let response: DocumentSymbolResponse = serde_json::from_value(
            self.send_request("textDocument/documentSymbol", serde_json::to_value(params)?).await?
        )?;

        // Flatten hierarchical symbols
        let mut symbols = Vec::new();
        match response {
            DocumentSymbolResponse::Nested(syms) => {
                self.flatten_symbols(syms, &mut symbols, None);
            }
            DocumentSymbolResponse::Flat(syms) => {
                symbols.extend(syms.into_iter().map(DocumentSymbol::from));
            }
        }
        Ok(symbols)
    }

    fn flatten_symbols(
        &self,
        symbols: Vec<lsp_types::DocumentSymbol>,
        out: &mut Vec<DocumentSymbol>,
        parent: Option<String>,
    ) {
        for sym in symbols {
            let mut ds = DocumentSymbol::from(sym);
            ds.parent = parent.clone();
            let name = ds.name.clone();
            out.push(ds);
            // Recursively flatten children using a local helper
            Self::flatten_local_symbols(out, &name);
        }
    }

    fn flatten_local_symbols(out: &mut Vec<DocumentSymbol>, parent: &str) {
        // Find the last added symbol that matches parent and flatten its children
        if let Some(idx) = out.iter().rposition(|s| s.name == parent) {
            let children = out[idx].children.clone();
            for mut child in children {
                child.parent = Some(parent.to_string());
                let child_name = child.name.clone();
                out.push(child);
                if !out.last().unwrap().children.is_empty() {
                    Self::flatten_local_symbols(out, &child_name);
                }
            }
        }
    }

    /// Get definition location for a symbol at position
    pub async fn goto_definition(&self, file_path: &Path, position: Position) -> Result<Vec<Location>> {
        let uri = Self::path_to_uri(file_path)?;

        let params = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: TextDocumentIdentifier { uri },
                position,
            },
            work_done_progress_params: WorkDoneProgressParams::default(),
            partial_result_params: Default::default(),
        };

        let response: GotoDefinitionResponse = serde_json::from_value(
            self.send_request("textDocument/definition", serde_json::to_value(params)?).await?
        )?;

        Ok(match response {
            GotoDefinitionResponse::Scalar(loc) => vec![loc],
            GotoDefinitionResponse::Array(locs) => locs,
            GotoDefinitionResponse::Link(links) => links.into_iter().map(|l| Location {
                uri: l.target_uri,
                range: l.target_range,
            }).collect(),
        })
    }

    /// Get references for a symbol at position
    pub async fn references(&self, file_path: &Path, position: Position) -> Result<Vec<Location>> {
        let uri = Self::path_to_uri(file_path)?;

        let params = ReferenceParams {
            text_document_position: lsp_types::TextDocumentPositionParams {
                text_document: TextDocumentIdentifier { uri },
                position,
            },
            context: ReferenceContext {
                include_declaration: true,
            },
            work_done_progress_params: WorkDoneProgressParams::default(),
            partial_result_params: Default::default(),
        };

        let response: Option<Vec<Location>> = serde_json::from_value(
            self.send_request("textDocument/references", serde_json::to_value(params)?).await?
        )?;

        Ok(response.unwrap_or_default())
    }

    /// Get implementations for a symbol at position
    pub async fn implementations(&self, file_path: &Path, position: Position) -> Result<Vec<Location>> {
        let uri = Self::path_to_uri(file_path)?;

        let params = GotoImplementationParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: TextDocumentIdentifier { uri },
                position,
            },
            work_done_progress_params: WorkDoneProgressParams::default(),
            partial_result_params: Default::default(),
        };

        let response: Option<Vec<Location>> = serde_json::from_value(
            self.send_request("textDocument/implementation", serde_json::to_value(params)?).await?
        )?;

        Ok(response.unwrap_or_default())
    }

    /// Prepare call hierarchy for a symbol
    pub async fn prepare_call_hierarchy(&self, file_path: &Path, position: Position) -> Result<Vec<CallHierarchyItem>> {
        let uri = Self::path_to_uri(file_path)?;

        let params = CallHierarchyPrepareParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: TextDocumentIdentifier { uri },
                position,
            },
            work_done_progress_params: WorkDoneProgressParams::default(),
        };

        let response: Option<Vec<CallHierarchyItem>> = serde_json::from_value(
            self.send_request("textDocument/prepareCallHierarchy", serde_json::to_value(params)?).await?
        )?;

        Ok(response.unwrap_or_default())
    }

    /// Get incoming calls for a call hierarchy item
    pub async fn incoming_calls(&self, item: CallHierarchyItem) -> Result<Vec<CallHierarchyIncomingCall>> {
        let params = CallHierarchyIncomingCallsParams {
            item,
            work_done_progress_params: WorkDoneProgressParams::default(),
            partial_result_params: Default::default(),
        };

        let response: Option<Vec<CallHierarchyIncomingCall>> = serde_json::from_value(
            self.send_request("callHierarchy/incomingCalls", serde_json::to_value(params)?).await?
        )?;

        Ok(response.unwrap_or_default())
    }

    /// Get outgoing calls for a call hierarchy item
    pub async fn outgoing_calls(&self, item: CallHierarchyItem) -> Result<Vec<CallHierarchyOutgoingCall>> {
        let params = CallHierarchyOutgoingCallsParams {
            item,
            work_done_progress_params: WorkDoneProgressParams::default(),
            partial_result_params: Default::default(),
        };

        let response: Option<Vec<CallHierarchyOutgoingCall>> = serde_json::from_value(
            self.send_request("callHierarchy/outgoingCalls", serde_json::to_value(params)?).await?
        )?;

        Ok(response.unwrap_or_default())
    }

    /// Shutdown the LSP connection
    pub async fn shutdown(&mut self) -> Result<()> {
        if self.initialized.load(std::sync::atomic::Ordering::SeqCst) {
            self.send_request("shutdown", json!({})).await?;
            self.send_notification("exit", json!({})).await?;
        }

        if let Some(mut process) = self.process.lock().await.take() {
            let _ = process.kill().await;
        }
        Ok(())
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        // Best effort cleanup
        if let Ok(mut process) = self.process.try_lock() {
            if let Some(mut child) = process.take() {
                let _ = child.start_kill();
            }
        }
    }
}

/// Simplified document symbol for CodeGraph integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSymbol {
    pub name: String,
    pub kind: SymbolKind,
    pub range: Range,
    pub selection_range: Range,
    pub detail: Option<String>,
    pub children: Vec<DocumentSymbol>,
    pub parent: Option<String>,
}

impl From<lsp_types::DocumentSymbol> for DocumentSymbol {
    fn from(sym: lsp_types::DocumentSymbol) -> Self {
        Self {
            name: sym.name,
            kind: sym.kind,
            range: sym.range,
            selection_range: sym.selection_range,
            detail: sym.detail,
            children: sym.children.unwrap_or_default().into_iter().map(Into::into).collect(),
            parent: None,
        }
    }
}

impl From<lsp_types::SymbolInformation> for DocumentSymbol {
    fn from(sym: lsp_types::SymbolInformation) -> Self {
        Self {
            name: sym.name,
            kind: sym.kind,
            range: sym.location.range,
            selection_range: sym.location.range,
            detail: None,
            children: Vec::new(),
            parent: None,
        }
    }
}

/// LSP Manager - handles lifecycle and provides high-level API
pub struct LspManager {
    client: Option<LspClient>,
    workspace_root: PathBuf,
    file_versions: std::sync::Mutex<HashMap<PathBuf, i32>>,
}

impl LspManager {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            client: None,
            workspace_root,
            file_versions: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Start the LSP client
    pub async fn start(&mut self) -> Result<()> {
        let mut client = LspClient::new(self.workspace_root.clone());
        client.start().await?;
        self.client = Some(client);
        Ok(())
    }

    /// Get the LSP client if available
    pub fn client(&self) -> Option<&LspClient> {
        self.client.as_ref()
    }

    /// Get mutable LSP client
    pub fn client_mut(&mut self) -> Option<&mut LspClient> {
        self.client.as_mut()
    }

    /// Check if LSP is available and initialized
    pub fn is_available(&self) -> bool {
        self.client.as_ref().map(|c| c.is_initialized()).unwrap_or(false)
    }

    /// Open or update a file in the LSP
    pub async fn sync_file(&self, file_path: &Path, content: &str) -> Result<()> {
        if let Some(client) = &self.client {
            let mut versions = self.file_versions.lock().unwrap();
            let version = versions.entry(file_path.to_path_buf()).or_insert(0);
            *version += 1;

            if *version == 1 {
                client.did_open(file_path, content).await?;
            } else {
                client.did_change(file_path, content, *version).await?;
            }
        }
        Ok(())
    }

    /// Get semantic symbols for a file (LSP + Tree-sitter fallback)
    pub async fn get_semantic_symbols(&self, file_path: &Path) -> Result<Vec<DocumentSymbol>> {
        if let Some(client) = &self.client {
            if client.is_initialized() {
                return client.document_symbols(file_path).await;
            }
        }
        Ok(Vec::new()) // Fallback handled by caller
    }

    /// Get definition locations
    pub async fn get_definition(&self, file_path: &Path, position: Position) -> Result<Vec<Location>> {
        if let Some(client) = &self.client {
            if client.is_initialized() {
                return client.goto_definition(file_path, position).await;
            }
        }
        Ok(Vec::new())
    }

    /// Get references
    pub async fn get_references(&self, file_path: &Path, position: Position) -> Result<Vec<Location>> {
        if let Some(client) = &self.client {
            if client.is_initialized() {
                return client.references(file_path, position).await;
            }
        }
        Ok(Vec::new())
    }

    /// Get implementations
    pub async fn get_implementations(&self, file_path: &Path, position: Position) -> Result<Vec<Location>> {
        if let Some(client) = &self.client {
            if client.is_initialized() {
                return client.implementations(file_path, position).await;
            }
        }
        Ok(Vec::new())
    }

    /// Get call hierarchy (incoming/outgoing)
    pub async fn get_call_hierarchy(&self, file_path: &Path, position: Position) -> Result<(Vec<CallHierarchyIncomingCall>, Vec<CallHierarchyOutgoingCall>)> {
        if let Some(client) = &self.client {
            if client.is_initialized() {
                let items = client.prepare_call_hierarchy(file_path, position).await?;
                if let Some(item) = items.first() {
                    let incoming = client.incoming_calls(item.clone()).await?;
                    let outgoing = client.outgoing_calls(item.clone()).await?;
                    return Ok((incoming, outgoing));
                }
            }
        }
        Ok((Vec::new(), Vec::new()))
    }

    /// Shutdown the LSP client
    pub async fn shutdown(&mut self) -> Result<()> {
        if let Some(mut client) = self.client.take() {
            client.shutdown().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_lsp_manager_creation() {
        let temp_dir = env::temp_dir().join("hercules_lsp_test");
        let _ = std::fs::create_dir_all(&temp_dir);
        let manager = LspManager::new(temp_dir.clone());
        assert!(!manager.is_available());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}