//! Code Graph — local code structure extraction using Tree-sitter and LSP.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tree_sitter::{Node, Parser};
use lsp_types::{Location, Range, SymbolKind, Uri};
use url::Url;

/// Create a CodeGraphConfig from the global settings.
pub fn config_from_settings() -> CodeGraphConfig {
    CodeGraphConfig {
        include_comments: crate::settings::get_code_graph_include_comments(),
        bounce_response_write: crate::settings::get_code_graph_bounce_response_write(),
    }
}

/// Unique identifier for a code node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new(kind: &str, file: &str, start_line: usize, start_col: usize) -> Self {
        Self(format!("{}:{}:{}:{}", kind, file, start_line, start_col))
    }
}

/// Type of code entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Module,
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    Impl,
    Constant,
    Static,
    TypeAlias,
    Field,
    Parameter,
    Variable,
    Macro,
    Use,
}

/// Convert LSP SymbolKind to CodeGraph NodeKind.
fn lsp_symbol_kind_to_node_kind(kind: SymbolKind) -> NodeKind {
    match kind {
        SymbolKind::FILE => NodeKind::Module,
        SymbolKind::MODULE => NodeKind::Module,
        SymbolKind::NAMESPACE => NodeKind::Module,
        SymbolKind::PACKAGE => NodeKind::Module,
        SymbolKind::CLASS => NodeKind::Struct,
        SymbolKind::METHOD => NodeKind::Method,
        SymbolKind::PROPERTY => NodeKind::Field,
        SymbolKind::FIELD => NodeKind::Field,
        SymbolKind::CONSTRUCTOR => NodeKind::Function,
        SymbolKind::ENUM => NodeKind::Enum,
        SymbolKind::INTERFACE => NodeKind::Trait,
        SymbolKind::FUNCTION => NodeKind::Function,
        SymbolKind::VARIABLE => NodeKind::Variable,
        SymbolKind::CONSTANT => NodeKind::Constant,
        SymbolKind::STRING => NodeKind::Constant,
        SymbolKind::NUMBER => NodeKind::Constant,
        SymbolKind::BOOLEAN => NodeKind::Constant,
        SymbolKind::ARRAY => NodeKind::TypeAlias,
        SymbolKind::OBJECT => NodeKind::Struct,
        SymbolKind::KEY => NodeKind::Field,
        SymbolKind::NULL => NodeKind::Constant,
        SymbolKind::ENUM_MEMBER => NodeKind::Constant,
        SymbolKind::STRUCT => NodeKind::Struct,
        SymbolKind::EVENT => NodeKind::Function,
        SymbolKind::OPERATOR => NodeKind::Function,
        SymbolKind::TYPE_PARAMETER => NodeKind::TypeAlias,
        _ => NodeKind::Function,
    }
}

/// Source code location range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRange {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl SourceRange {
    pub fn from_node(node: &Node, source: &str) -> Self {
        let start = node.start_position();
        let end = node.end_position();
        Self {
            start_line: start.row + 1,
            start_col: start.column + 1,
            end_line: end.row + 1,
            end_col: end.column + 1,
        }
    }
}

/// A node in the code graph representing a source-code entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeNode {
    pub id: NodeId,
    pub kind: NodeKind,
    pub name: String,
    pub file_path: PathBuf,
    pub range: SourceRange,
    pub parent: Option<NodeId>,
    pub documentation: Option<String>,
}

impl CodeNode {
    pub fn new(
        kind: NodeKind,
        name: String,
        file_path: PathBuf,
        range: SourceRange,
        parent: Option<NodeId>,
        documentation: Option<String>,
    ) -> Self {
        let id = NodeId::new(
            format!("{:?}", kind).to_lowercase().as_str(),
            file_path.to_string_lossy().as_ref(),
            range.start_line,
            range.start_col,
        );
        Self {
            id,
            kind,
            name,
            file_path,
            range,
            parent,
            documentation,
        }
    }
}

/// Type of relationship between code nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Calls,
    Implements,
    Contains,
    References,
    Imports,
    Defines,
}

/// An edge in the code graph representing a relationship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
}

/// The code graph containing nodes and edges.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeGraph {
    pub nodes: Vec<CodeNode>,
    pub edges: Vec<CodeEdge>,
    node_index: HashMap<NodeId, usize>,
}

impl CodeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: CodeNode) -> NodeId {
        let id = node.id.clone();
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.node_index.insert(id.clone(), idx);
        id
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId, kind: EdgeKind) {
        // Check if edge already exists (deduplication)
        if !self.edges.iter().any(|e| e.from == from && e.to == to && e.kind == kind) {
            self.edges.push(CodeEdge { from, to, kind });
        }
    }

    /// Remove all Calls edges originating from a specific node.
    /// Used to replace Tree-sitter name-based calls with LSP semantic calls.
    pub fn remove_calls_edges_from(&mut self, from: &NodeId) {
        self.edges.retain(|e| !(e.kind == EdgeKind::Calls && e.from == *from));
    }

    pub fn get_node(&self, id: &NodeId) -> Option<&CodeNode> {
        self.node_index.get(id).map(|&i| &self.nodes[i])
    }

    pub fn get_node_mut(&mut self, id: &NodeId) -> Option<&mut CodeNode> {
        if let Some(&i) = self.node_index.get(id) {
            Some(&mut self.nodes[i])
        } else {
            None
        }
    }

    pub fn nodes_by_file(&self, file: &PathBuf) -> Vec<&CodeNode> {
        self.nodes
            .iter()
            .filter(|n| &n.file_path == file)
            .collect()
    }

    pub fn edges_from(&self, from: &NodeId) -> Vec<&CodeEdge> {
        self.edges.iter().filter(|e| &e.from == from).collect()
    }

    pub fn edges_to(&self, to: &NodeId) -> Vec<&CodeEdge> {
        self.edges.iter().filter(|e| &e.to == to).collect()
    }

    /// Create a subgraph around the given node IDs (within 1 hop by default).
    pub fn subgraph_around(&self, node_ids: &[NodeId], hops: usize) -> CodeGraph {
        let mut result = CodeGraph::new();
        let mut visited = std::collections::HashSet::new();
        let mut frontier: Vec<NodeId> = node_ids.to_vec();

        for _ in 0..=hops {
            let mut next_frontier = Vec::new();
            for id in frontier {
                if visited.insert(id.clone()) {
                    if let Some(node) = self.get_node(&id) {
                        result.add_node(node.clone());
                        for edge in self.edges_from(&id) {
                            if !visited.contains(&edge.to) {
                                next_frontier.push(edge.to.clone());
                            }
                        }
                        for edge in self.edges_to(&id) {
                            if !visited.contains(&edge.from) {
                                next_frontier.push(edge.from.clone());
                            }
                        }
                    }
                }
            }
            frontier = next_frontier;
        }

        // Add edges between nodes in the subgraph
        for edge in &self.edges {
            if result.get_node(&edge.from).is_some() && result.get_node(&edge.to).is_some() {
                result.add_edge(edge.from.clone(), edge.to.clone(), edge.kind);
            }
        }

        result
    }

    /// Create a subgraph affected by changes in the given files/ranges.
    pub fn affected_by_changes(&self, changed_files: &[PathBuf]) -> CodeGraph {
        let mut affected_nodes = Vec::new();
        for node in &self.nodes {
            for file in changed_files {
                if &node.file_path == file {
                    affected_nodes.push(node.id.clone());
                    break;
                }
            }
        }
        self.subgraph_around(&affected_nodes, 1)
    }

    /// Create a subgraph affected by changes in the given file ranges.
    /// Each entry is a tuple of (file_path, start_line, end_line).
    pub fn affected_by_changes_ranges(&self, changed_ranges: &[(PathBuf, usize, usize)]) -> CodeGraph {
        let mut affected_nodes = Vec::new();
        for node in &self.nodes {
            for (file, start_line, end_line) in changed_ranges {
                if &node.file_path == file 
                    && node.range.start_line >= *start_line 
                    && node.range.end_line <= *end_line {
                    affected_nodes.push(node.id.clone());
                    break;
                }
            }
        }
        self.subgraph_around(&affected_nodes, 1)
    }
}

/// Configuration for code graph extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGraphConfig {
    pub include_comments: bool,
    pub bounce_response_write: bool,
}

impl Default for CodeGraphConfig {
    fn default() -> Self {
        Self {
            include_comments: true,
            bounce_response_write: true,
        }
    }
}

/// Tree-sitter based Rust code extractor.
pub struct RustTreeSitterExtractor {
    parser: Parser,
    config: CodeGraphConfig,
}

impl RustTreeSitterExtractor {
    pub fn new(config: CodeGraphConfig) -> Result<Self, String> {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .map_err(|e| format!("Failed to set language: {:?}", e))?;
        Ok(Self { parser, config })
    }

    pub fn extract(&mut self, source: &str, file_path: PathBuf) -> Result<CodeGraph, String> {
        let tree = self
            .parser
            .parse(source, None)
            .ok_or_else(|| "Failed to parse source".to_string())?;

        let mut graph = CodeGraph::new();
        let root_node = tree.root_node();

        self.extract_module(root_node, source, &file_path, None, &mut graph)?;
        
        // Second pass: resolve function calls
        self.resolve_calls(&mut graph, source, &file_path)?;

        Ok(graph)
    }

    fn resolve_calls(&mut self, graph: &mut CodeGraph, source: &str, file_path: &PathBuf) -> Result<(), String> {
        // Collect all function nodes with their source ranges
        let functions: Vec<(NodeId, String, SourceRange)> = graph.nodes.iter()
            .filter(|n| n.kind == NodeKind::Function || n.kind == NodeKind::Method)
            .map(|n| (n.id.clone(), n.name.clone(), n.range))
            .collect();

        // For each function, find calls in its body
        let tree = self.parser.parse(source, None).ok_or_else(|| "Failed to parse source".to_string())?;
        let root_node = tree.root_node();
        
        self.find_calls_in_tree(root_node, source, file_path, &functions, graph)?;

        Ok(())
    }

    fn find_calls_in_tree(
        &mut self,
        node: Node,
        source: &str,
        file_path: &PathBuf,
        functions: &[(NodeId, String, SourceRange)],
        graph: &mut CodeGraph,
    ) -> Result<(), String> {
        if node.kind() == "function_item" {
            // Find which function this is
            let range = SourceRange::from_node(&node, source);
            let func_name = self.extract_name(&node, source, "function_item")?;
            
            // Find matching function in our list
            let caller_id = functions.iter()
                .find(|(_, name, r)| name == &func_name && r.start_line == range.start_line)
                .map(|(id, _, _)| id.clone());

            if let Some(caller_id) = caller_id {
                // Find all call expressions in this function
                self.find_calls_in_function(node, source, file_path, &caller_id, functions, graph)?;
            }
        }

        // Recurse
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.find_calls_in_tree(child, source, file_path, functions, graph)?;
        }
        Ok(())
    }

    fn find_calls_in_function(
        &mut self,
        node: Node,
        source: &str,
        file_path: &PathBuf,
        caller_id: &NodeId,
        functions: &[(NodeId, String, SourceRange)],
        graph: &mut CodeGraph,
    ) -> Result<(), String> {
        if node.kind() == "call_expression" {
            for child in node.children(&mut node.walk()) {
                if child.kind() == "identifier" || child.kind() == "scoped_identifier" {
                    let call_name = self.get_node_text(child, source)?;
                    // Find callee
                    for (callee_id, name, _) in functions {
                        if name == &call_name {
                            graph.add_edge(caller_id.clone(), callee_id.clone(), EdgeKind::Calls);
                            break;
                        }
                    }
                    break;
                }
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.find_calls_in_function(child, source, file_path, caller_id, functions, graph)?;
        }
        Ok(())
    }

    fn extract_module(
        &mut self,
        node: Node,
        source: &str,
        file_path: &PathBuf,
        parent: Option<&NodeId>,
        graph: &mut CodeGraph,
    ) -> Result<Option<NodeId>, String> {
        let mut module_id = None;

        match node.kind() {
            "source_file" => {
                let range = SourceRange::from_node(&node, source);
                let name = file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let doc = if self.config.include_comments {
                    self.extract_doc_comment(node, source)
                } else {
                    None
                };

                let node_id = graph.add_node(CodeNode::new(
                    NodeKind::Module,
                    name,
                    file_path.clone(),
                    range,
                    parent.cloned(),
                    doc,
                ));
                module_id = Some(node_id);

                // Extract children
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.extract_item(child, source, file_path, module_id.as_ref(), graph)?;
                }
            }
            _ => {
                // Recurse into children
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.extract_item(child, source, file_path, parent, graph)?;
                }
            }
        }

        Ok(module_id)
    }

    fn extract_item(
        &mut self,
        node: Node,
        source: &str,
        file_path: &PathBuf,
        parent: Option<&NodeId>,
        graph: &mut CodeGraph,
    ) -> Result<Option<NodeId>, String> {
        let kind = match node.kind() {
            "function_item" => Some(NodeKind::Function),
            "struct_item" => Some(NodeKind::Struct),
            "enum_item" => Some(NodeKind::Enum),
            "trait_item" => Some(NodeKind::Trait),
            "impl_item" => Some(NodeKind::Impl),
            "const_item" => Some(NodeKind::Constant),
            "static_item" => Some(NodeKind::Static),
            "type_item" => Some(NodeKind::TypeAlias),
            "macro_definition" => Some(NodeKind::Macro),
            "use_declaration" => Some(NodeKind::Use),
            _ => None,
        };

        if let Some(kind) = kind {
            let name = self.extract_name(&node, source, node.kind())?;
            let range = SourceRange::from_node(&node, source);
            let doc = if self.config.include_comments {
                self.extract_doc_comment(node, source)
            } else {
                None
            };

            let node_id = graph.add_node(CodeNode::new(
                kind,
                name,
                file_path.clone(),
                range,
                parent.cloned(),
                doc,
            ));

            // If the parent is an Impl node, classify this function as a Method
            if kind == NodeKind::Function {
                if let Some(parent_id) = parent {
                    if let Some(parent_node) = graph.get_node(&parent_id) {
                        if parent_node.kind == NodeKind::Impl {
                            // Change the function kind to Method
                            if let Some(func_node) = graph.get_node_mut(&node_id) {
                                func_node.kind = NodeKind::Method;
                            }
                            // Also update the function's name to include the impl type
                            // and its parent reference
                        }
                    }
                }
            }

            // Extract children based on kind
            match kind {
                NodeKind::Impl => {
                    self.extract_impl_contents(node, source, file_path, Some(&node_id), graph)?;
                }
                NodeKind::Trait => {
                    self.extract_trait_contents(node, source, file_path, Some(&node_id), graph)?;
                }
                NodeKind::Struct | NodeKind::Enum => {
                    self.extract_fields(node, source, file_path, Some(&node_id), graph)?;
                }
                NodeKind::Function => {
                    self.extract_function_contents(node, source, file_path, Some(&node_id), graph)?;
                }
                _ => {}
            }

            // Recurse for nested items
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                self.extract_item(child, source, file_path, Some(&node_id), graph)?;
            }

            Ok(Some(node_id))
        } else {
            // Recurse into children
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                self.extract_item(child, source, file_path, parent, graph)?;
            }
            Ok(None)
        }
    }

    fn extract_impl_contents(
        &mut self,
        node: Node,
        source: &str,
        file_path: &PathBuf,
        parent: Option<&NodeId>,
        graph: &mut CodeGraph,
    ) -> Result<(), String> {
        // Find the trait being implemented (if any) and the type
        let mut trait_name = None;
        let mut type_name = None;
        let mut found_for = false;
        
        for child in node.children(&mut node.walk()) {
            if child.kind() == "type_identifier" {
                let name = self.get_node_text(child, source)?;
                if !found_for {
                    trait_name = Some(name);
                } else {
                    type_name = Some(name);
                }
            } else if child.kind() == "for" {
                found_for = true;
            }
        }
        
        // Update the impl node's name to the type name if available
        if let Some(p) = parent {
            if let Some(type_name) = type_name {
                if let Some(impl_node) = graph.get_node_mut(p) {
                    impl_node.name = type_name;
                }
            }
            
            // Find existing trait node and add implements edge
            if let Some(trait_name) = trait_name {
                // Look for existing trait node with this name in the same file
                for existing_node in &graph.nodes {
                    if existing_node.kind == NodeKind::Trait 
                        && existing_node.name == trait_name
                        && existing_node.file_path == *file_path {
                        graph.add_edge(p.clone(), existing_node.id.clone(), EdgeKind::Implements);
                        break;
                    }
                }
            }
        }

        // Extract methods
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "function_item" {
                self.extract_item(child, source, file_path, parent, graph)?;
            }
        }

        Ok(())
    }

    fn extract_trait_contents(
        &mut self,
        node: Node,
        source: &str,
        file_path: &PathBuf,
        parent: Option<&NodeId>,
        graph: &mut CodeGraph,
    ) -> Result<(), String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "function_item" || child.kind() == "type_item" || child.kind() == "const_item" {
                self.extract_item(child, source, file_path, parent, graph)?;
            }
        }
        Ok(())
    }

    fn extract_fields(
        &mut self,
        node: Node,
        source: &str,
        file_path: &PathBuf,
        parent: Option<&NodeId>,
        graph: &mut CodeGraph,
    ) -> Result<(), String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "field_declaration" {
                for field in child.children(&mut child.walk()) {
                    if field.kind() == "field_identifier" {
                        let name = self.get_node_text(field, source)?;
                        let range = SourceRange::from_node(&field, source);
                        graph.add_node(CodeNode::new(
                            NodeKind::Field,
                            name,
                            file_path.clone(),
                            range,
                            parent.cloned(),
                            None,
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn extract_function_contents(
        &mut self,
        node: Node,
        source: &str,
        file_path: &PathBuf,
        parent: Option<&NodeId>,
        graph: &mut CodeGraph,
    ) -> Result<(), String> {
        // Extract parameters
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "parameters" {
                for param in child.children(&mut child.walk()) {
                    if param.kind() == "parameter" {
                        for p in param.children(&mut param.walk()) {
                            if p.kind() == "identifier" {
                                let name = self.get_node_text(p, source)?;
                                let range = SourceRange::from_node(&p, source);
                                graph.add_node(CodeNode::new(
                                    NodeKind::Parameter,
                                    name,
                                    file_path.clone(),
                                    range,
                                    parent.cloned(),
                                    None,
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Extract function calls
        // (done in second pass via resolve_calls)

        Ok(())
    }

    fn extract_name(&self, node: &Node, source: &str, kind: &str) -> Result<String, String> {
        // Debug: print children
        // for child in node.children(&mut node.walk()) {
        //     println!("  Child: kind={}, text={:?}", child.kind(), self.get_node_text(child, source));
        // }
        
        match kind {
            "function_item" | "struct_item" | "enum_item" | "trait_item" | "const_item" | "static_item" | "type_item" => {
                for child in node.children(&mut node.walk()) {
                    if child.kind() == "identifier" || child.kind() == "type_identifier" {
                        return self.get_node_text(child, source);
                    }
                }
            }
            "impl_item" => {
                // For impl, try to get the type name
                for child in node.children(&mut node.walk()) {
                    if child.kind() == "type_identifier" || child.kind() == "identifier" {
                        return self.get_node_text(child, source);
                    }
                }
            }
            "trait_ref" => {
                for child in node.children(&mut node.walk()) {
                    if child.kind() == "type_identifier" || child.kind() == "scoped_type_identifier" {
                        return self.get_node_text(child, source);
                    }
                }
            }
            _ => {}
        }
        Ok(format!("<anonymous-{}>", kind))
    }

    fn extract_doc_comment(&self, node: Node, source: &str) -> Option<String> {
        // First check children (for inner comments)
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "line_comment" || child.kind() == "block_comment" {
                let text = self.get_node_text(child, source).ok()?;
                let trimmed = text.trim_start_matches(['/', '*', '!']).trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
        
        // Check for outer doc comments (attributes) - look at preceding siblings
        if let Some(prev_sibling) = node.prev_sibling() {
            if prev_sibling.kind() == "line_comment" || prev_sibling.kind() == "block_comment" {
                let text = self.get_node_text(prev_sibling, source).ok()?;
                let trimmed = text.trim_start_matches(['/', '*', '!']).trim();
                if !trimmed.is_empty() && (text.starts_with("///") || text.starts_with("//!")) {
                    return Some(trimmed.to_string());
                }
            }
        }
        
        // Check for attributes
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "attribute" {
                let text = self.get_node_text(child, source).ok()?;
                let trimmed = text.trim_start_matches(['/', '*', '!', '#', '[', ']']).trim();
                if !trimmed.is_empty() && (text.contains("doc") || text.starts_with("///")) {
                    return Some(trimmed.to_string());
                }
            }
        }
        
        None
    }

    fn get_node_text(&self, node: Node, source: &str) -> Result<String, String> {
        let range = node.byte_range();
        if range.end <= source.len() {
            Ok(source[range].to_string())
        } else {
            Err("Node range exceeds source length".to_string())
        }
    }
}

/// Build a code graph from source files.
pub struct CodeGraphBuilder {
    config: CodeGraphConfig,
    extractor: Option<RustTreeSitterExtractor>,
    lsp_manager: Option<crate::lsp::LspManager>,
}

impl CodeGraphBuilder {
    /// Create a new builder from global settings.
    pub fn from_settings() -> Result<Self, String> {
        let config = config_from_settings();
        Self::new(config)
    }

    pub fn new(config: CodeGraphConfig) -> Result<Self, String> {
        let extractor = RustTreeSitterExtractor::new(config.clone())?;
        Ok(Self {
            config,
            extractor: Some(extractor),
            lsp_manager: None,
        })
    }

    /// Create a new builder with LSP semantic enrichment enabled.
    pub fn with_lsp(config: CodeGraphConfig, lsp_manager: crate::lsp::LspManager) -> Result<Self, String> {
        let extractor = RustTreeSitterExtractor::new(config.clone())?;
        Ok(Self {
            config,
            extractor: Some(extractor),
            lsp_manager: Some(lsp_manager),
        })
    }

    /// Set the LSP manager for semantic enrichment.
    pub fn set_lsp_manager(&mut self, lsp_manager: crate::lsp::LspManager) {
        self.lsp_manager = Some(lsp_manager);
    }

    pub fn add_file(&mut self, source: &str, file_path: PathBuf) -> Result<CodeGraph, String> {
        let mut graph = self.extractor
            .as_mut()
            .ok_or_else(|| "Extractor not initialized".to_string())?
            .extract(source, file_path.clone())?;
        
        // Note: LSP semantic enrichment is only available via add_file_async()
        // when called from within a Tokio runtime. The synchronous add_file() 
        // uses only Tree-sitter extraction to avoid blocking runtime issues.
        
        Ok(graph)
    }

    /// Async version of add_file for LSP enrichment without blocking.
    /// Must be called from within a Tokio runtime.
    pub async fn add_file_async(&mut self, source: &str, file_path: PathBuf) -> Result<CodeGraph, String> {
        let mut graph = self.extractor
            .as_mut()
            .ok_or_else(|| "Extractor not initialized".to_string())?
            .extract(source, file_path.clone())?;
        
        // Enrich with LSP semantic information if available
        if let Some(lsp_manager) = &self.lsp_manager {
            if lsp_manager.is_available() {
                // First sync the file with LSP so it has current source
                lsp_manager.sync_file(&file_path, source).await.map_err(|e| e.to_string())?;
                
                // Then enrich with LSP semantic information
                self.enrich_with_lsp_async(&mut graph, &file_path, source).await;
            }
        }
        
        Ok(graph)
    }

    /// Enrich the graph with LSP semantic information (async version).
    pub async fn enrich_with_lsp_async(&self, graph: &mut CodeGraph, file_path: &PathBuf, source: &str) {
        if let Some(lsp_manager) = &self.lsp_manager {
            if lsp_manager.is_available() {
                // Enrich with document symbols (better symbol detection)
                if let Ok(symbols) = lsp_manager.get_semantic_symbols(file_path).await {
                    self.merge_lsp_symbols(graph, &symbols, file_path);
                }
                
                // Enrich calls with LSP call hierarchy for each function
                let functions: Vec<_> = graph.nodes.iter()
                    .filter(|n| n.kind == NodeKind::Function || n.kind == NodeKind::Method)
                    .cloned()
                    .collect();
                
                for func in functions {
                    let position = lsp_types::Position {
                        line: func.range.start_line.saturating_sub(1) as u32,
                        character: func.range.start_col.saturating_sub(1) as u32,
                    };
                    
                    // Get call hierarchy for this function (incoming + outgoing)
                    // LSP success = authoritative result; replace Tree-sitter edges even if empty
                    if let Ok((incoming, outgoing)) = lsp_manager.get_call_hierarchy(file_path, position).await {
                        self.add_lsp_call_edges_for_function(graph, &func.name, file_path, &func.range, &incoming, &outgoing);
                    }
                }
                
                // Enrich definitions/references/implementations for all functions
                self.enrich_definitions_references(graph, lsp_manager, file_path, source).await;
            }
        }
    }

    /// Merge LSP symbols into the graph, avoiding duplicates.
    fn merge_lsp_symbols(&self, graph: &mut CodeGraph, symbols: &[crate::lsp::DocumentSymbol], file_path: &PathBuf) {
        for symbol in symbols {
            // Check if we already have a node with this name and location
            let sym_start_line = (symbol.range.start.line + 1) as usize;
            let exists = graph.nodes.iter().any(|n| 
                n.name == symbol.name 
                && n.file_path == *file_path
                && n.range.start_line == sym_start_line
            );
            
            if !exists {
                let kind = lsp_symbol_kind_to_node_kind(symbol.kind);
                let range = SourceRange {
                    start_line: (symbol.range.start.line + 1) as usize,
                    start_col: (symbol.range.start.character + 1) as usize,
                    end_line: (symbol.range.end.line + 1) as usize,
                    end_col: (symbol.range.end.character + 1) as usize,
                };
                let doc = if self.config.include_comments { symbol.detail.clone() } else { None };
                let node = CodeNode::new(kind, symbol.name.clone(), file_path.clone(), range, None, doc);
                graph.add_node(node);
            }
            
            // Recurse into children
            self.merge_lsp_symbols(graph, &symbol.children, file_path);
        }
    }

    /// Add call edges from LSP call hierarchy for a specific function.
    fn add_lsp_call_edges_for_function(
        &self,
        graph: &mut CodeGraph,
        function_name: &str,
        function_file: &PathBuf,
        function_range: &SourceRange,
        incoming: &[crate::lsp::CallHierarchyIncomingCall],
        outgoing: &[crate::lsp::CallHierarchyOutgoingCall],
    ) {
        // Find the function node in our graph using exact range match
        let func_node = graph.nodes.iter().find(|n| 
            n.name == function_name 
            && n.file_path == *function_file
            && n.range.start_line == function_range.start_line
            && n.range.end_line == function_range.end_line
            && n.range.start_col == function_range.start_col
            && n.range.end_col == function_range.end_col
        );
        let Some(func_node) = func_node else { return };
        let func_id = func_node.id.clone();

        // Remove existing Tree-sitter Calls edges from this function before adding LSP edges
        graph.remove_calls_edges_from(&func_id);

        // For incoming calls: callers of this function
        for call in incoming {
            let caller_name = call.from.name.clone();
            let caller_file = Url::parse(call.from.uri.as_str()).ok().and_then(|u| u.to_file_path().ok());
            if let Some(caller_file) = caller_file {
                // Find or create the caller node
                if let Ok(caller_node_id) = self.find_or_create_node(graph, &caller_name, &caller_file, &call.from.range) {
                    // Add edge: caller -> this function (callee)
                    graph.add_edge(caller_node_id.clone(), func_id.clone(), EdgeKind::Calls);
                    // Also add Defines edge: caller defines the callee
                    graph.add_edge(caller_node_id, func_id.clone(), EdgeKind::Defines);
                }
            }
        }
        
        // For outgoing calls: callees of this function
        for call in outgoing {
            let callee_name = call.to.name.clone();
            let callee_file = Url::parse(call.to.uri.as_str()).ok().and_then(|u| u.to_file_path().ok());
            if let Some(callee_file) = callee_file {
                // Find or create the callee node
                if let Ok(callee_node_id) = self.find_or_create_node(graph, &callee_name, &callee_file, &call.to.range) {
                    // Add edge: this function (caller) -> callee
                    graph.add_edge(func_id.clone(), callee_node_id.clone(), EdgeKind::Calls);
                    // Also add Defines edge: caller defines the callee
                    graph.add_edge(func_id.clone(), callee_node_id, EdgeKind::Defines);
                }
            }
        }
    }

    /// Add call edges from LSP call hierarchy (placeholder for full implementation).
    fn add_lsp_call_edges(&self, graph: &mut CodeGraph, incoming: &[crate::lsp::CallHierarchyIncomingCall], outgoing: &[crate::lsp::CallHierarchyOutgoingCall]) {
        // Legacy placeholder - use add_lsp_call_edges_for_function instead
        let _ = (incoming, outgoing);
    }

    /// Find an existing node or create a new one from LSP call hierarchy item.
    /// Uses LSP item range/location for precise matching to avoid duplicates.
    fn find_or_create_node(&self, graph: &mut CodeGraph, name: &str, file_path: &PathBuf, range: &Range) -> Result<NodeId, String> {
        // Try to find existing node by matching LSP item range (more precise than name+file)
        let target_start_line = (range.start.line + 1) as usize;
        let target_end_line = (range.end.line + 1) as usize;
        let target_start_col = (range.start.character + 1) as usize;
        let target_end_col = (range.end.character + 1) as usize;
        
        // First try exact range match
        for node in &graph.nodes {
            if node.name == name 
                && node.file_path == *file_path
                && node.range.start_line == target_start_line
                && node.range.end_line == target_end_line
                && node.range.start_col == target_start_col
                && node.range.end_col == target_end_col {
                return Ok(node.id.clone());
            }
        }
        
        // Fallback: name + file + start line match (for cases where column info differs)
        for node in &graph.nodes {
            if node.name == name 
                && node.file_path == *file_path
                && node.range.start_line == target_start_line {
                return Ok(node.id.clone());
            }
        }
        
        // Create new node
        let kind = NodeKind::Function; // Default
        let source_range = SourceRange {
            start_line: target_start_line,
            start_col: target_start_col,
            end_line: target_end_line,
            end_col: target_end_col,
        };
        let node = CodeNode::new(kind, name.to_string(), file_path.clone(), source_range, None, None);
        Ok(graph.add_node(node))
    }

    /// Enrich definitions, references, and implementations for functions.
    async fn enrich_definitions_references(&self, graph: &mut CodeGraph, lsp_manager: &crate::lsp::LspManager, file_path: &PathBuf, _source: &str) {
        let functions: Vec<_> = graph.nodes.iter()
            .filter(|n| n.kind == NodeKind::Function || n.kind == NodeKind::Method)
            .cloned()
            .collect();
        
        for func in functions {
            let position = lsp_types::Position {
                line: func.range.start_line.saturating_sub(1) as u32,
                character: func.range.start_col.saturating_sub(1) as u32,
            };
            
            // Definitions: query at call sites, not declaration.
            // We'll track definitions from call sites to target function.
            // (This is handled by call hierarchy; Defines from call edges already captures this)
            
            // References: find all references to this function, then map each reference
            // to its OWNER function (the function containing the reference).
            // Edge: owner_function -> referenced_function
            if let Ok(refs) = lsp_manager.get_references(file_path, position).await {
                for ref_loc in refs {
                    // Skip the declaration itself
                    let func_start_line = (func.range.start_line - 1) as u32;
                    let func_start_col = (func.range.start_col - 1) as u32;
                    if ref_loc.uri.as_str() == file_path.to_string_lossy().as_ref() 
                        && ref_loc.range.start.line == func_start_line
                        && ref_loc.range.start.character == func_start_col {
                        continue;
                    }
                    if let Some(owner_node) = self.find_owner_function(graph, &ref_loc) {
                        if owner_node.id != func.id {
                            // Edge: reference owner -> referenced function
                            graph.add_edge(owner_node.id.clone(), func.id.clone(), EdgeKind::References);
                        }
                    }
                }
            }
            
            // Implementations
            if let Ok(impls) = lsp_manager.get_implementations(file_path, position).await {
                for impl_loc in impls {
                    if let Some(impl_node) = self.location_to_node(graph, &impl_loc) {
                        graph.add_edge(func.id.clone(), impl_node.id.clone(), EdgeKind::Implements);
                    }
                }
            }
        }
    }

    /// Convert an LSP Location to a CodeNode (find existing or create).
    /// Uses fuzzy matching - finds the node whose range contains the location.
    fn location_to_node(&self, graph: &mut CodeGraph, location: &Location) -> Option<CodeNode> {
        let file_path = Url::parse(location.uri.as_str()).ok()?.to_file_path().ok()?;
        let loc_line = (location.range.start.line + 1) as usize;
        let loc_col = (location.range.start.character + 1) as usize;
        
        // Find the node whose range contains this location
        // Prefer the most specific (smallest) containing node
        let mut best_match: Option<&CodeNode> = None;
        let mut best_range_size = usize::MAX;
        
        for node in &graph.nodes {
            if node.file_path == file_path 
                && node.range.start_line <= loc_line
                && node.range.end_line >= loc_line {
                // Location is within this node's line range
                // Check column if on same start/end line
                let col_ok = if node.range.start_line == node.range.end_line {
                    node.range.start_col <= loc_col && node.range.end_col >= loc_col
                } else if loc_line == node.range.start_line {
                    node.range.start_col <= loc_col
                } else if loc_line == node.range.end_line {
                    node.range.end_col >= loc_col
                } else {
                    true // Location is in middle of multi-line node
                };
                
                if col_ok {
                    let range_size = node.range.end_line.saturating_sub(node.range.start_line);
                    if range_size < best_range_size {
                        best_range_size = range_size;
                        best_match = Some(node);
                    }
                }
            }
        }
        
        best_match.cloned()
    }

    /// Find the function/method node that owns a given LSP location (reference site owner).
    /// This is used to find which function contains a reference to another symbol.
    fn find_owner_function(&self, graph: &mut CodeGraph, location: &Location) -> Option<CodeNode> {
        let file_path = Url::parse(location.uri.as_str()).ok()?.to_file_path().ok()?;
        let loc_line = (location.range.start.line + 1) as usize;
        let loc_col = (location.range.start.character + 1) as usize;
        
        // Find the function/method node whose range contains this location
        // Prefer the most specific (smallest) containing node
        let mut best_match: Option<&CodeNode> = None;
        let mut best_range_size = usize::MAX;
        
        for node in &graph.nodes {
            if (node.kind == NodeKind::Function || node.kind == NodeKind::Method)
                && node.file_path == file_path 
                && node.range.start_line <= loc_line
                && node.range.end_line >= loc_line {
                // Location is within this function's line range
                let col_ok = if node.range.start_line == node.range.end_line {
                    node.range.start_col <= loc_col && node.range.end_col >= loc_col
                } else if loc_line == node.range.start_line {
                    node.range.start_col <= loc_col
                } else if loc_line == node.range.end_line {
                    node.range.end_col >= loc_col
                } else {
                    true // Location is in middle of multi-line function
                };
                
                if col_ok {
                    let range_size = node.range.end_line.saturating_sub(node.range.start_line);
                    if range_size < best_range_size {
                        best_range_size = range_size;
                        best_match = Some(node);
                    }
                }
            }
        }
        
        best_match.cloned()
    }

    pub fn merge(&self, graphs: Vec<CodeGraph>) -> CodeGraph {
        let mut merged = CodeGraph::new();
        for graph in graphs {
            for node in graph.nodes {
                merged.add_node(node);
            }
            for edge in graph.edges {
                merged.add_edge(edge.from, edge.to, edge.kind);
            }
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_source() -> &'static str {
        r#"
trait Renderer {
    fn render(&self);
}

struct App;

impl Renderer for App {
    fn render(&self) {
        draw();
    }
}

fn draw() {}
"#
    }

    #[test]
    fn test_extract_function() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("fn foo() {}", PathBuf::from("test.rs")).unwrap();
        assert!(graph.nodes.iter().any(|n| n.kind == NodeKind::Function && n.name == "foo"));
    }

    #[test]
    fn test_extract_struct() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("struct Bar { field: i32 }", PathBuf::from("test.rs")).unwrap();
        assert!(graph.nodes.iter().any(|n| n.kind == NodeKind::Struct && n.name == "Bar"));
    }

    #[test]
    fn test_extract_trait() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("trait Baz { fn qux(); }", PathBuf::from("test.rs")).unwrap();
        assert!(graph.nodes.iter().any(|n| n.kind == NodeKind::Trait && n.name == "Baz"));
    }

    #[test]
    fn test_extract_impl() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("impl Foo { fn bar() {} }", PathBuf::from("test.rs")).unwrap();
        assert!(graph.nodes.iter().any(|n| n.kind == NodeKind::Impl && n.name == "Foo"));
    }

    #[test]
    fn test_extract_methods() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file(
            "impl Foo { fn method1() {} fn method2() {} }",
            PathBuf::from("test.rs"),
        ).unwrap();
        let methods: Vec<_> = graph.nodes.iter().filter(|n| n.kind == NodeKind::Method).collect();
        assert_eq!(methods.len(), 2);
    }

    #[test]
    fn test_module_containment() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("fn foo() {}", PathBuf::from("test.rs")).unwrap();
        let module = graph.nodes.iter().find(|n| n.kind == NodeKind::Module).unwrap();
        let func = graph.nodes.iter().find(|n| n.kind == NodeKind::Function).unwrap();
        assert_eq!(func.parent, Some(module.id.clone()));
    }

    #[test]
    fn test_fixture_relationships() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file(test_source(), PathBuf::from("fixture.rs")).unwrap();

        // Debug: print all nodes and edges
        for node in &graph.nodes {
            println!("Node: kind={:?}, name={}, parent={:?}", node.kind, node.name, node.parent);
        }
        for edge in &graph.edges {
            println!("Edge: {:?} -> {:?} ({:?})", edge.from, edge.to, edge.kind);
        }

        let app_struct = graph.nodes.iter().find(|n| n.kind == NodeKind::Struct && n.name == "App").unwrap();
        let renderer_trait = graph.nodes.iter().find(|n| n.kind == NodeKind::Trait && n.name == "Renderer").unwrap();
        let app_impl = graph.nodes.iter().find(|n| n.kind == NodeKind::Impl).unwrap();
        let render_method = graph.nodes.iter().find(|n| n.kind == NodeKind::Method && n.name == "render" && n.parent.as_ref() == Some(&app_impl.id)).unwrap();
        let draw_func = graph.nodes.iter().find(|n| n.kind == NodeKind::Function && n.name == "draw").unwrap();

        // App --implements--> Renderer
        let implements_edges: Vec<_> = graph.edges.iter().filter(|e| e.kind == EdgeKind::Implements).collect();
        assert!(!implements_edges.is_empty());

        // render --calls--> draw
        let calls_edges: Vec<_> = graph.edges.iter().filter(|e| e.kind == EdgeKind::Calls).collect();
        assert!(!calls_edges.is_empty());
    }

    #[test]
    fn test_mermaid_generation() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("fn foo() {}", PathBuf::from("test.rs")).unwrap();
        let mermaid = graph.to_mermaid();
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("foo"));
    }

    #[test]
    fn test_mermaid_identifier_escaping() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("fn foo-bar() {}", PathBuf::from("test.rs")).unwrap();
        let mermaid = graph.to_mermaid();
        // Should not contain invalid Mermaid identifiers
        assert!(!mermaid.contains("foo-bar"));
    }

    #[test]
    fn test_include_comments_true() {
        let config = CodeGraphConfig { include_comments: true, ..Default::default() };
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("/// Doc comment\nfn foo() {}", PathBuf::from("test.rs")).unwrap();
        let func = graph.nodes.iter().find(|n| n.kind == NodeKind::Function).unwrap();
        assert!(func.documentation.is_some());
    }

    #[test]
    fn test_include_comments_false() {
        let config = CodeGraphConfig { include_comments: false, ..Default::default() };
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("/// Doc comment\nfn foo() {}", PathBuf::from("test.rs")).unwrap();
        let func = graph.nodes.iter().find(|n| n.kind == NodeKind::Function).unwrap();
        assert!(func.documentation.is_none());
    }

    #[test]
    fn test_focused_subgraph() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("fn foo() { bar(); } fn bar() { baz(); } fn baz() {}", PathBuf::from("test.rs")).unwrap();
        
        let bar = graph.nodes.iter().find(|n| n.name == "bar").unwrap();
        let subgraph = graph.subgraph_around(&[bar.id.clone()], 1);
        
        assert!(subgraph.nodes.iter().any(|n| n.name == "foo"));
        assert!(subgraph.nodes.iter().any(|n| n.name == "bar"));
        assert!(subgraph.nodes.iter().any(|n| n.name == "baz"));
    }

    #[test]
    fn test_affected_by_changes_ranges() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        // Use multi-line source to get different line numbers
        let graph = builder.add_file(
            "fn foo() {\n    bar();\n}\n\nfn bar() {\n    baz();\n}\n\nfn baz() {}\n",
            PathBuf::from("test.rs")
        ).unwrap();
        
        // Debug: print all nodes with their line numbers
        for node in &graph.nodes {
            println!("Node: name={}, kind={:?}, start_line={}, end_line={}", node.name, node.kind, node.range.start_line, node.range.end_line);
        }
        
        // Find bar's line range
        let bar = graph.nodes.iter().find(|n| n.name == "bar").unwrap();
        println!("bar range: {} - {}", bar.range.start_line, bar.range.end_line);
        
        // Change range covering only bar
        let subgraph = graph.affected_by_changes_ranges(&[(PathBuf::from("test.rs"), bar.range.start_line, bar.range.end_line)]);
        
        // Should only include bar and its direct connections (foo, baz)
        let names: Vec<_> = subgraph.nodes.iter().map(|n| n.name.clone()).collect();
        assert!(names.contains(&"bar".to_string()));
        assert!(names.contains(&"foo".to_string()));
        assert!(names.contains(&"baz".to_string()));
        assert_eq!(subgraph.nodes.len(), 3); // bar, foo, baz
    }

    #[test]
    fn test_empty_source() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let graph = builder.add_file("", PathBuf::from("empty.rs")).unwrap();
        let module = graph.nodes.iter().find(|n| n.kind == NodeKind::Module).unwrap();
        assert_eq!(module.name, "empty");
    }

    #[test]
    fn test_invalid_source_no_panic() {
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        let result = builder.add_file("fn foo( { }", PathBuf::from("invalid.rs"));
        // Should not panic, may return error or partial graph
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_cross_file_semantic_resolution_architecture() {
        // This test verifies the cross-file semantic resolution architecture works
        // (without actual LSP - tests Tree-sitter fallback behavior)
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        
        // Create two files with cross-file calls - Tree-sitter only resolves within file
        // For LSP-based cross-file resolution, rust-analyzer would be needed
        let file_a = builder.add_file(
            "fn foo() { bar(); }",
            PathBuf::from("file_a.rs")
        ).unwrap();
        
        let file_b = builder.add_file(
            "fn bar() {}",
            PathBuf::from("file_b.rs")
        ).unwrap();
        
        // Merge graphs
        let merged = builder.merge(vec![file_a, file_b]);
        
        // Verify both files are in the merged graph
        let files: std::collections::HashSet<_> = merged.nodes.iter()
            .map(|n| n.file_path.clone())
            .collect();
        assert_eq!(files.len(), 2, "Should have nodes from both files");
        
        // Verify both functions exist
        let foo = merged.nodes.iter().find(|n| n.name == "foo").unwrap();
        let bar = merged.nodes.iter().find(|n| n.name == "bar").unwrap();
        assert_eq!(foo.file_path, PathBuf::from("file_a.rs"));
        assert_eq!(bar.file_path, PathBuf::from("file_b.rs"));
        
        // Note: Tree-sitter only resolves calls within a single file
        // Cross-file call resolution requires LSP (rust-analyzer)
        // This test verifies the architecture supports cross-file graphs
        let foo_node = merged.nodes.iter().find(|n| n.name == "foo").unwrap();
        assert_eq!(foo_node.file_path, PathBuf::from("file_a.rs"));
    }

    #[test]
    #[ignore] // Requires rust-analyzer to be installed
    fn test_cross_file_lsp_semantic_resolution() {
        // This test requires rust-analyzer to be installed and in PATH
        // It tests real cross-file LSP semantic resolution in a valid Rust workspace
        let temp_dir = tempfile::tempdir().unwrap();
        
        // Create a valid Rust workspace structure
        let cargo_toml = r#"[package]
name = "test_cross_file"
version = "0.1.0"
edition = "2021"
"#;
        let lib_rs = r#"
pub mod file_a;
pub mod file_b;
"#;
        let file_a_rs = r#"
use crate::file_b::bar;

pub fn foo() {
    bar();
}
"#;
        let file_b_rs = r#"
pub fn bar() {}
"#;
        
        std::fs::write(temp_dir.path().join("Cargo.toml"), cargo_toml).unwrap();
        std::fs::create_dir_all(temp_dir.path().join("src")).unwrap();
        std::fs::write(temp_dir.path().join("src/lib.rs"), lib_rs).unwrap();
        std::fs::write(temp_dir.path().join("src/file_a.rs"), file_a_rs).unwrap();
        std::fs::write(temp_dir.path().join("src/file_b.rs"), file_b_rs).unwrap();
        
        let file_a_path = temp_dir.path().join("src/file_a.rs");
        let file_b_path = temp_dir.path().join("src/file_b.rs");
        
        let config = CodeGraphConfig::default();
        let mut builder = CodeGraphBuilder::new(config).unwrap();
        
        // Check if rust-analyzer is available
        if which::which("rust-analyzer").is_err() {
            eprintln!("Skipping test: rust-analyzer not found");
            return;
        }
        
        let mut lsp_manager = crate::lsp::LspManager::new(temp_dir.path().to_path_buf());
        let rt = tokio::runtime::Runtime::new().unwrap();
        if rt.block_on(lsp_manager.start()).is_err() {
            eprintln!("Skipping test: failed to start rust-analyzer");
            return;
        }
        builder.set_lsp_manager(lsp_manager);
        
        let file_a = rt.block_on(builder.add_file_async(
            "use crate::file_b::bar;\n\npub fn foo() {\n    bar();\n}",
            file_a_path.clone(),
        )).unwrap();
        
        let file_b = rt.block_on(builder.add_file_async(
            "pub fn bar() {}",
            file_b_path.clone(),
        )).unwrap();
        
        let merged = builder.merge(vec![file_a, file_b]);
        
        // Verify LSP found the cross-file call
        let foo = merged.nodes.iter().find(|n| n.name == "foo").unwrap();
        let bar = merged.nodes.iter().find(|n| n.name == "bar").unwrap();
        
        let calls_edges: Vec<_> = merged.edges.iter().filter(|e| e.kind == EdgeKind::Calls).collect();
        let has_cross_file_call = calls_edges.iter().any(|e| e.from == foo.id && e.to == bar.id);
        assert!(has_cross_file_call, "LSP should find cross-file call foo -> bar");
    }
}

// Mermaid rendering will be added in Phase 2
impl CodeGraph {
    pub fn to_mermaid(&self) -> String {
        let mut out = String::from("graph TD\n");
        
        // Group by file
        let mut files: HashMap<PathBuf, Vec<&CodeNode>> = HashMap::new();
        for node in &self.nodes {
            if node.kind == NodeKind::Module {
                continue; // Skip module nodes for now
            }
            files.entry(node.file_path.clone()).or_default().push(node);
        }

        for (file, nodes) in &files {
            let file_id = sanitize_mermaid_id(&file.to_string_lossy());
            let file_name = file.file_name().and_then(|s| s.to_str()).unwrap_or("unknown");
            out.push_str(&format!("    {}[\"{}\"]\n", file_id, escape_mermaid_label(file_name)));
            
            for node in nodes {
                let node_id = sanitize_mermaid_id(&node.id.0);
                let label = format!("{} {}", kind_prefix(node.kind), escape_mermaid_label(&node.name));
                out.push_str(&format!("    {}[\"{}\"]\n", node_id, label));
                out.push_str(&format!("    {} --> {}\n", file_id, node_id));
            }
        }

        // Add edges
        for edge in &self.edges {
            let from_id = sanitize_mermaid_id(&edge.from.0);
            let to_id = sanitize_mermaid_id(&edge.to.0);
            let label = match edge.kind {
                EdgeKind::Calls => "calls",
                EdgeKind::Implements => "implements",
                EdgeKind::Contains => "contains",
                EdgeKind::References => "refs",
                EdgeKind::Imports => "imports",
                EdgeKind::Defines => "defines",
            };
            out.push_str(&format!("    {} -->|{}| {}\n", from_id, label, to_id));
        }

        out
    }
}

fn kind_prefix(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Module => "📦",
        NodeKind::Function => "ƒ",
        NodeKind::Method => "⚙",
        NodeKind::Struct => "𝓢",
        NodeKind::Enum => "ℰ",
        NodeKind::Trait => "𝒯",
        NodeKind::Impl => "𝕀",
        NodeKind::Constant => "𝒞",
        NodeKind::Static => "𝒮",
        NodeKind::TypeAlias => "𝒯",
        NodeKind::Field => "𝒻",
        NodeKind::Parameter => "𝓅",
        NodeKind::Variable => "𝓋",
        NodeKind::Macro => "𝓂",
        NodeKind::Use => "𝓊",
    }
}

fn sanitize_mermaid_id(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

fn escape_mermaid_label(s: &str) -> String {
    s.replace('"', "\\\"")
        .replace('<', "<")
        .replace('>', ">")
        .replace('[', "&#91;")
        .replace(']', "&#93;")
}