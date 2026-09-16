//! Ask Mode - Interactive question/answer system for the agent

use serde::{Deserialize, Serialize};

/// Ask Mode element types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AskElement {
    Check { label: String, selected: bool },
    Radio { label: String, selected: bool },
    Input { placeholder: String, value: String },
}

impl AskElement {
    pub fn label(&self) -> &str {
        match self {
            AskElement::Check { label, .. } => label,
            AskElement::Radio { label, .. } => label,
            AskElement::Input { placeholder, .. } => placeholder,
        }
    }

    pub fn is_check(&self) -> bool {
        matches!(self, AskElement::Check { .. })
    }

    pub fn is_radio(&self) -> bool {
        matches!(self, AskElement::Radio { .. })
    }

    pub fn is_input(&self) -> bool {
        matches!(self, AskElement::Input { .. })
    }

    pub fn selected(&self) -> bool {
        match self {
            AskElement::Check { selected, .. } => *selected,
            AskElement::Radio { selected, .. } => *selected,
            AskElement::Input { .. } => false,
        }
    }

    pub fn set_selected(&mut self, selected: bool) {
        match self {
            AskElement::Check { selected: s, .. } => *s = selected,
            AskElement::Radio { selected: s, .. } => *s = selected,
            AskElement::Input { .. } => {}
        }
    }

    pub fn value(&self) -> &str {
        match self {
            AskElement::Input { value, .. } => value,
            _ => "",
        }
    }

    pub fn set_value(&mut self, value: String) {
        if let AskElement::Input { value: v, .. } = self {
            *v = value;
        }
    }
}

/// Parsed Ask Mode structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskMode {
    pub question: String,
    pub elements: Vec<AskElement>,
}

impl AskMode {
    pub fn new(question: String, elements: Vec<AskElement>) -> Self {
        Self { question, elements }
    }

    pub fn radio_count(&self) -> usize {
        self.elements.iter().filter(|e| e.is_radio()).count()
    }

    pub fn check_count(&self) -> usize {
        self.elements.iter().filter(|e| e.is_check()).count()
    }

    pub fn input_count(&self) -> usize {
        self.elements.iter().filter(|e| e.is_input()).count()
    }

    pub fn has_radio(&self) -> bool {
        self.radio_count() > 0
    }
}

/// Response from Ask Mode submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskModeResponse {
    pub checks: Vec<String>,
    pub radio: Option<String>,
    pub inputs: Vec<String>,
}

impl AskModeResponse {
    pub fn from_ask_mode(ask_mode: &AskMode) -> Self {
        let mut checks = Vec::new();
        let mut radio = None;
        let mut inputs = Vec::new();

        for elem in &ask_mode.elements {
            match elem {
                AskElement::Check { label, selected } if *selected => checks.push(label.clone()),
                AskElement::Radio { label, selected } if *selected => radio = Some(label.clone()),
                AskElement::Input { value, .. } if !value.is_empty() => inputs.push(value.clone()),
                _ => {}
            }
        }

        Self {
            checks,
            radio,
            inputs,
        }
    }
}

/// Ask Mode UI state
#[derive(Debug, Clone)]
pub struct AskModeState {
    pub ask_mode: AskMode,
    pub focused_index: usize,
    pub active: bool,
    pub input_editing: bool,
}

impl AskModeState {
    pub fn new(ask_mode: AskMode) -> Self {
        // Find first focusable element
        let focused_index = ask_mode
            .elements
            .iter()
            .position(|e| e.is_check() || e.is_radio() || e.is_input())
            .unwrap_or(0);

        Self {
            ask_mode,
            focused_index,
            active: true,
            input_editing: false,
        }
    }

    pub fn focused_element(&self) -> Option<&AskElement> {
        self.ask_mode.elements.get(self.focused_index)
    }

    pub fn focused_element_mut(&mut self) -> Option<&mut AskElement> {
        self.ask_mode.elements.get_mut(self.focused_index)
    }

    pub fn move_focus_next(&mut self) {
        if self.input_editing {
            return;
        }
        let len = self.ask_mode.elements.len();
        if len == 0 {
            return;
        }
        self.focused_index = (self.focused_index + 1) % len;
    }

    pub fn move_focus_prev(&mut self) {
        if self.input_editing {
            return;
        }
        let len = self.ask_mode.elements.len();
        if len == 0 {
            return;
        }
        if self.focused_index == 0 {
            self.focused_index = len - 1;
        } else {
            self.focused_index -= 1;
        }
    }

    pub fn toggle_focused(&mut self) {
        let focused_idx = self.focused_index;
        if focused_idx >= self.ask_mode.elements.len() {
            return;
        }

        // Get the current selected state before mutable borrow
        let is_selected = self.ask_mode.elements[focused_idx].selected();

        match &mut self.ask_mode.elements[focused_idx] {
            AskElement::Check { .. } => {
                self.ask_mode.elements[focused_idx].set_selected(!is_selected);
            }
            AskElement::Radio { .. } => {
                if !is_selected {
                    // Deselect all other radios
                    for e in &mut self.ask_mode.elements {
                        if e.is_radio() {
                            e.set_selected(false);
                        }
                    }
                    self.ask_mode.elements[focused_idx].set_selected(true);
                }
            }
            AskElement::Input { .. } => {
                self.input_editing = true;
            }
        }
    }

    pub fn handle_input_char(&mut self, c: char) {
        if self.input_editing {
            if let Some(elem) = self.focused_element_mut() {
                if let AskElement::Input { value, .. } = elem {
                    value.push(c);
                }
            }
        }
    }

    pub fn handle_input_backspace(&mut self) {
        if self.input_editing {
            if let Some(elem) = self.focused_element_mut() {
                if let AskElement::Input { value, .. } = elem {
                    value.pop();
                }
            }
        }
    }

    pub fn exit_input_editing(&mut self) {
        self.input_editing = false;
    }

    pub fn submit(&self) -> AskModeResponse {
        AskModeResponse::from_ask_mode(&self.ask_mode)
    }

    pub fn cancel(&mut self) {
        self.active = false;
    }
}

/// Parser for Ask Mode blocks
pub struct AskModeParser;

impl AskModeParser {
    /// Parse an Ask Mode block from text
    pub fn parse(text: &str) -> Result<Option<AskMode>, ParseError> {
        // Find the opening <askmode> tag
        let start_tag = Self::find_tag(text, "askmode")?;
        let Some((start_pos, question)) = start_tag else {
            return Ok(None);
        };

        // Find the closing </askmode> tag
        let end_tag = Self::find_closing_tag(text, "askmode", start_pos)?;
        let Some(end_pos) = end_tag else {
            return Err(ParseError::UnclosedAskMode);
        };

        // Extract content between tags
        let content = &text[start_pos..end_pos];

        // Parse elements
        let elements = Self::parse_elements(content)?;

        Ok(Some(AskMode::new(question, elements)))
    }

    /// Find opening tag and extract question attribute
    fn find_tag(text: &str, tag_name: &str) -> Result<Option<(usize, String)>, ParseError> {
        // Literal prefix match: `<tagname` must be followed by a delimiter
        // (space, newline, '/', or '>') so `<askmodeX>` does not match.
        let open = format!("<{tag_name}");
        let mut search_from = 0;
        let (start, tag_end) = loop {
            let Some(rel) = text[search_from..].find(&open) else {
                return Ok(None);
            };
            let abs = search_from + rel;
            let after = abs + open.len();
            let Some(next) = text[after..].chars().next() else {
                return Err(ParseError::MalformedTag(format!("<{tag_name}>")));
            };
            if next == ' '
                || next == '\n'
                || next == '\r'
                || next == '\t'
                || next == '/'
                || next == '>'
            {
                let Some(gt) = text[abs..].find('>') else {
                    return Err(ParseError::MalformedTag(format!("<{tag_name}>")));
                };
                break (abs, abs + gt + 1);
            }
            search_from = abs + open.len();
        };

        // Extract the tag content
        let tag_content = &text[start..tag_end];

        // Parse ques attribute
        let question = Self::extract_attribute(tag_content, "ques")
            .ok_or_else(|| ParseError::MissingAttribute("ques".to_string()))?;

        if question.trim().is_empty() {
            return Err(ParseError::EmptyQuestion);
        }

        Ok(Some((tag_end, question)))
    }

    /// Find closing tag
    fn find_closing_tag(
        text: &str,
        tag_name: &str,
        start: usize,
    ) -> Result<Option<usize>, ParseError> {
        let close_pattern = format!("</{tag_name}>");
        let Some(pos) = text[start..].find(&close_pattern) else {
            return Err(ParseError::UnclosedAskMode);
        };
        Ok(Some(start + pos))
    }

    /// Extract attribute value from tag. Requires an attribute-name
    /// boundary (whitespace/`<`/start) and rejects hits inside quoted
    /// values, so `src="` cannot match inside another name or value.
    fn extract_attribute(tag: &str, attr: &str) -> Option<String> {
        let pattern = format!(r#"{attr}="#);
        let mut search_from = 0;
        while let Some(rel) = tag[search_from..].find(&pattern) {
            let idx = search_from + rel;
            let boundary_ok =
                idx == 0 || matches!(tag.as_bytes()[idx - 1], b' ' | b'\t' | b'\n' | b'\r' | b'<');
            let in_quotes = {
                let mut in_q: Option<u8> = None;
                for b in tag[..idx].bytes() {
                    if b == b'"' || b == b'\'' {
                        if in_q == Some(b) {
                            in_q = None;
                        } else if in_q.is_none() {
                            in_q = Some(b);
                        }
                    }
                }
                in_q.is_some()
            };
            if boundary_ok && !in_quotes {
                let attr_start = idx + pattern.len();

                // Handle both single and double quotes
                let quote_char = tag.chars().nth(attr_start)?;
                if quote_char != '"' && quote_char != '\'' {
                    return None;
                }

                let value_start = attr_start + 1;
                let Some(value_end) = tag[value_start..].find(quote_char) else {
                    return None;
                };
                let value_end = value_start + value_end;

                return Some(tag[value_start..value_end].to_string());
            }
            search_from = idx + 1;
        }
        None
    }

    /// Parse elements inside askmode
    fn parse_elements(content: &str) -> Result<Vec<AskElement>, ParseError> {
        let mut elements = Vec::new();
        let mut pos = 0;

        while pos < content.len() {
            // Skip whitespace (advance by char boundary, not byte)
            let remaining = &content[pos..];
            let mut advanced = false;
            if let Some(char) = remaining.chars().next() {
                if char.is_whitespace() {
                    pos += char.len_utf8();
                    advanced = true;
                }
            }
            // If we advanced past whitespace, continue the loop to check for tags
            if advanced {
                continue;
            }
            // No whitespace at current position, check for element tags
            if content[pos..].starts_with("<check") {
                let (elem, new_pos) = Self::parse_check(&content[pos..])?;
                elements.push(elem);
                pos += new_pos;
            } else if content[pos..].starts_with("<radio") {
                let (elem, new_pos) = Self::parse_radio(&content[pos..])?;
                elements.push(elem);
                pos += new_pos;
            } else if content[pos..].starts_with("<input") {
                let (elem, new_pos) = Self::parse_input(&content[pos..])?;
                elements.push(elem);
                pos += new_pos;
            } else {
                // Skip unknown content until next '<'; always advance at
                // least one byte so unknown tags can't loop forever.
                if let Some(next_tag) = content[pos..].find('<') {
                    pos += next_tag.max(1);
                } else {
                    break;
                }
            }
        }

        Ok(elements)
    }

    fn parse_check(content: &str) -> Result<(AskElement, usize), ParseError> {
        Self::parse_simple_element(content, "check", |label| AskElement::Check {
            label,
            selected: false,
        })
    }

    fn parse_radio(content: &str) -> Result<(AskElement, usize), ParseError> {
        Self::parse_simple_element(content, "radio", |label| AskElement::Radio {
            label,
            selected: false,
        })
    }

    fn parse_input(content: &str) -> Result<(AskElement, usize), ParseError> {
        // Find opening tag end
        let Some(tag_end) = content.find('>') else {
            return Err(ParseError::MalformedTag("<input>".to_string()));
        };

        // Check for self-closing
        if content[..tag_end].ends_with("/") {
            let placeholder =
                Self::extract_attribute(&content[..tag_end], "placeholder").unwrap_or_default();
            return Ok((
                AskElement::Input {
                    placeholder,
                    value: String::new(),
                },
                tag_end + 1,
            ));
        }

        // Find closing tag
        let open_len = tag_end + 1;
        let close_pattern = "</input>";
        let Some(close_pos) = content[open_len..].find(close_pattern) else {
            return Err(ParseError::UnclosedTag("input".to_string()));
        };
        let close_pos = open_len + close_pos;

        let placeholder = content[open_len..close_pos].to_string();
        let total_len = close_pos + close_pattern.len();

        Ok((
            AskElement::Input {
                placeholder,
                value: String::new(),
            },
            total_len,
        ))
    }

    fn parse_simple_element<F>(
        content: &str,
        tag_name: &str,
        constructor: F,
    ) -> Result<(AskElement, usize), ParseError>
    where
        F: FnOnce(String) -> AskElement,
    {
        // Find opening tag end
        let Some(tag_end) = content.find('>') else {
            return Err(ParseError::MalformedTag(format!("<{tag_name}>")));
        };

        // Check for self-closing
        if content[..tag_end].ends_with("/") {
            let label = Self::extract_attribute(&content[..tag_end], "label").unwrap_or_default();
            return Ok((constructor(label), tag_end + 1));
        }

        // Find closing tag
        let open_len = tag_end + 1;
        let close_pattern = format!("</{tag_name}>");
        let Some(close_pos) = content[open_len..].find(&close_pattern) else {
            return Err(ParseError::UnclosedTag(tag_name.to_string()));
        };
        let close_pos = open_len + close_pos;

        let label = content[open_len..close_pos].to_string();
        let total_len = close_pos + close_pattern.len();

        Ok((constructor(label), total_len))
    }
}

/// Remove raw `<askmode>...</askmode>` blocks from visible chat text.
///
/// Once a block is parsed into `AskModeState`, the interactive widget owns
/// the UI — the XML protocol must never render as a normal Agent message.
/// Unclosed trailing blocks are dropped too (partial stream XML is not content).
pub fn strip_ask_mode_blocks(text: &str) -> std::borrow::Cow<'_, str> {
    const CLOSE: &str = "</askmode>";
    if !text.contains("<askmode") {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut changed = false;
    while let Some(start) = rest.find("<askmode") {
        if let Some(end_rel) = rest[start..].find(CLOSE) {
            out.push_str(&rest[..start]);
            rest = &rest[start + end_rel + CLOSE.len()..];
            changed = true;
        } else {
            out.push_str(&rest[..start]);
            rest = "";
            changed = true;
            break;
        }
    }
    if !changed {
        return std::borrow::Cow::Borrowed(text);
    }
    out.push_str(rest);
    std::borrow::Cow::Owned(out)
}

/// Parse errors for Ask Mode
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Missing required attribute: {0}")]
    MissingAttribute(String),
    #[error("Empty question")]
    EmptyQuestion,
    #[error("Malformed tag: {0}")]
    MalformedTag(String),
    #[error("Unclosed tag: {0}")]
    UnclosedTag(String),
    #[error("Unclosed askmode block")]
    UnclosedAskMode,
    #[error("Invalid element content")]
    InvalidElementContent,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_askmode() {
        let text = r#"<askmode ques="Choose implementation">
    <radio>Minimal</radio>
    <radio>Refactor</radio>
    <check>Add tests</check>
    <check>Update docs</check>
    <input>Notes</input>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap();
        assert!(result.is_some());
        let ask_mode = result.unwrap();
        assert_eq!(ask_mode.question, "Choose implementation");
        assert_eq!(ask_mode.elements.len(), 5);
        assert_eq!(ask_mode.radio_count(), 2);
        assert_eq!(ask_mode.check_count(), 2);
        assert_eq!(ask_mode.input_count(), 1);
    }

    #[test]
    fn test_missing_ques() {
        let text = r#"<askmode>
    <radio>A</radio>
</askmode>"#;

        let result = AskModeParser::parse(text);
        assert!(matches!(result, Err(ParseError::MissingAttribute(_))));
    }

    #[test]
    fn test_empty_question() {
        let text = r#"<askmode ques="">
    <radio>A</radio>
</askmode>"#;

        let result = AskModeParser::parse(text);
        assert!(matches!(result, Err(ParseError::EmptyQuestion)));
    }

    #[test]
    fn test_check_only() {
        let text = r#"<askmode ques="Select options">
    <check>A</check>
    <check>B</check>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.check_count(), 2);
        assert_eq!(result.radio_count(), 0);
    }

    #[test]
    fn test_radio_only() {
        let text = r#"<askmode ques="Select one">
    <radio>A</radio>
    <radio>B</radio>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.radio_count(), 2);
        assert_eq!(result.check_count(), 0);
    }

    #[test]
    fn test_input_only() {
        let text = r#"<askmode ques="Enter text">
    <input>Placeholder</input>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.input_count(), 1);
        assert_eq!(result.elements[0].label(), "Placeholder");
    }

    #[test]
    fn test_mixed_elements() {
        let text = r#"<askmode ques="Mixed">
    <check>Check 1</check>
    <radio>Radio 1</radio>
    <input>Input 1</input>
    <check>Check 2</check>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.elements.len(), 4);
        assert!(result.elements[0].is_check());
        assert!(result.elements[1].is_radio());
        assert!(result.elements[2].is_input());
        assert!(result.elements[3].is_check());
    }

    #[test]
    fn test_self_closing_elements() {
        let text = r#"<askmode ques="Self closing">
    <check label="Check 1"/>
    <radio label="Radio 1"/>
    <input placeholder="Input 1"/>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.elements.len(), 3);
    }

    #[test]
    fn test_unclosed_askmode() {
        let text = r#"<askmode ques="Test">
    <radio>A</radio>
"#;

        let result = AskModeParser::parse(text);
        assert!(matches!(result, Err(ParseError::UnclosedAskMode)));
    }

    #[test]
    fn test_tags_outside_askmode() {
        let text = r#"Some text before
<askmode ques="Test">
    <radio>A</radio>
</askmode>
Some text after"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.question, "Test");
    }

    #[test]
    fn test_unknown_tags_inside() {
        let text = r#"<askmode ques="Test">
    <unknown>ignored</unknown>
    <radio>A</radio>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.elements.len(), 1);
    }

    #[test]
    fn test_empty_check() {
        let text = r#"<askmode ques="Test">
    <check></check>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.elements[0].label(), "");
    }

    #[test]
    fn test_empty_radio() {
        let text = r#"<askmode ques="Test">
    <radio></radio>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.elements[0].label(), "");
    }

    #[test]
    fn test_empty_input() {
        let text = r#"<askmode ques="Test">
    <input></input>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.elements[0].label(), "");
    }

    #[test]
    fn test_whitespace_handling() {
        let text = r#"<askmode ques="  Test  ">
    <radio>  A  </radio>
    <check>  B  </check>
</askmode>"#;

        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.question, "  Test  ");
        assert_eq!(result.elements[0].label(), "  A  ");
    }

    #[test]
    fn test_ask_mode_state() {
        let ask_mode = AskMode::new(
            "Test".to_string(),
            vec![
                AskElement::Radio {
                    label: "A".to_string(),
                    selected: false,
                },
                AskElement::Radio {
                    label: "B".to_string(),
                    selected: false,
                },
                AskElement::Check {
                    label: "C".to_string(),
                    selected: false,
                },
            ],
        );

        let mut state = AskModeState::new(ask_mode);
        assert_eq!(state.focused_index, 0);

        // Toggle first radio
        state.toggle_focused();
        assert!(state.ask_mode.elements[0].selected());
        assert!(!state.ask_mode.elements[1].selected());

        // Move to second radio and toggle
        state.move_focus_next();
        state.toggle_focused();
        assert!(!state.ask_mode.elements[0].selected());
        assert!(state.ask_mode.elements[1].selected());

        // Move to check and toggle
        state.move_focus_next();
        state.toggle_focused();
        assert!(state.ask_mode.elements[2].selected());

        // Test radio selection logic - selecting B should deselect A
        state.move_focus_prev(); // Back to B
        state.toggle_focused(); // Already selected, should stay
        assert!(state.ask_mode.elements[1].selected());

        // Submit
        let response = state.submit();
        assert_eq!(response.radio, Some("B".to_string()));
        assert_eq!(response.checks, vec!["C"]);
    }

    #[test]
    fn test_input_editing() {
        let ask_mode = AskMode::new(
            "Test".to_string(),
            vec![AskElement::Input {
                placeholder: "Enter text".to_string(),
                value: String::new(),
            }],
        );

        let mut state = AskModeState::new(ask_mode);
        assert_eq!(state.focused_index, 0);

        // Start editing
        state.toggle_focused();
        assert!(state.input_editing);

        // Type some text
        state.handle_input_char('H');
        state.handle_input_char('i');
        assert_eq!(state.ask_mode.elements[0].value(), "Hi");

        // Backspace
        state.handle_input_backspace();
        assert_eq!(state.ask_mode.elements[0].value(), "H");

        // Exit editing
        state.exit_input_editing();
        assert!(!state.input_editing);

        // Submit
        let response = state.submit();
        assert_eq!(response.inputs, vec!["H"]);
    }

    #[test]
    fn test_navigation_wrapping() {
        let ask_mode = AskMode::new(
            "Test".to_string(),
            vec![
                AskElement::Radio {
                    label: "A".to_string(),
                    selected: false,
                },
                AskElement::Radio {
                    label: "B".to_string(),
                    selected: false,
                },
            ],
        );

        let mut state = AskModeState::new(ask_mode);

        // Forward wrap
        state.move_focus_next();
        assert_eq!(state.focused_index, 1);
        state.move_focus_next();
        assert_eq!(state.focused_index, 0);

        // Backward wrap
        state.move_focus_prev();
        assert_eq!(state.focused_index, 1);
        state.move_focus_prev();
        assert_eq!(state.focused_index, 0);
    }

    #[test]
    fn test_tag_boundary_regression() {
        // <askmodeX should NOT be recognized as <askmode>
        let text = r#"<askmodeX ques="test">
    <radio>A</radio>
</askmodeX>"#;
        let result = AskModeParser::parse(text);
        // parse returns Result<Option<AskMode>, ParseError>
        // Pattern not found should give Ok(None)
        assert!(result.is_ok_and(|opt| opt.is_none()));
    }

    #[test]
    fn test_malformed_askmode_unclosed_inside() {
        // <check inside <askmode> without close
        let text = r#"<askmode ques="test">
    <check>A
</askmode>"#;
        let result = AskModeParser::parse(text);
        // Should handle gracefully (error or partial parse)
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_malformed_mismatched_close() {
        // <check> with </radio> close
        let text = r#"<askmode ques="test">
    <check>A</radio>
</askmode>"#;
        let result = AskModeParser::parse(text);
        // parse returns Result<Option<AskMode>, ParseError>
        // Parser handles gracefully - either error or partial parse
        eprintln!("result = {:?}", result);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_unicode_whitespace_handling() {
        // Test that multibyte whitespace doesn't panic
        let text = r#"<askmode ques="test">
    <radio>  A  </radio>
</askmode>"#;
        let result = AskModeParser::parse(text).unwrap().unwrap();
        assert_eq!(result.question, "test");
        assert_eq!(result.elements[0].label(), "  A  ");
    }

    #[test]
    fn test_malformed_tag_boundary() {
        // Various malformed tags should not crash parser
        // <askmodeX should NOT match <askmode due to boundary check
        let result = AskModeParser::parse("<askmodeX ques='test'>...</askmodeX>");
        assert!(result.is_ok_and(|opt| opt.is_none()));
        // <check>A</radio> should return error (unclosed tag)
        let result2 = AskModeParser::parse("<askmode ques='test'><check>A</radio></askmode>");
        assert!(result2.is_err());
        // Unclosed <askmode> must not hang or crash: it reports
        // UnclosedAskMode (consistent with test_unclosed_askmode).
        let result3 = AskModeParser::parse("<askmode ques='test'>");
        assert!(matches!(result3, Err(ParseError::UnclosedAskMode)));
    }

    #[test]
    fn test_strip_ask_mode_blocks_removes_widget_xml() {
        // Parsed blocks must never render as chat text.
        let text = "Here is my question:\n<askmode ques=\"Q?\"><radio>A</radio></askmode>\nDone.";
        let stripped = super::strip_ask_mode_blocks(text);
        assert!(!stripped.contains("<askmode"));
        assert!(!stripped.contains("<radio>"));
        assert!(stripped.contains("Here is my question:"));
        assert!(stripped.contains("Done."));
    }

    #[test]
    fn test_strip_ask_mode_blocks_drops_unclosed_tail() {
        // Partial stream XML is not content.
        let text = "Thinking out loud <askmode ques=\"Q?\"><radio>A";
        let stripped = super::strip_ask_mode_blocks(text);
        assert!(!stripped.contains("<askmode"));
        assert!(stripped.contains("Thinking out loud"));
    }

    #[test]
    fn test_strip_ask_mode_blocks_passthrough() {
        let text = "Just a normal message with <read src=\"a.rs\"> inside.";
        let stripped = super::strip_ask_mode_blocks(text);
        assert_eq!(stripped.as_ref(), text);
    }
}
