use crate::{
    syntax::Syntax,
    tree_sitter::{Capture, Grammar, InactiveQueryCursor, Query, RopeInput},
    RopeSlice,
};
use helix_stdx::rope::RopeSliceExt;
use tree_house::TREE_SITTER_MATCH_LIMIT;

/// A query for determining when to apply "electric" indentation behaviors,
/// i.e. re-indenting the current line as soon as a keyword such as python's
/// `else` has been typed, instead of waiting for the next newline.
#[derive(Debug)]
pub struct ElectricQuery {
    query: Query,
    electric_keyword: Option<Capture>,
}

impl ElectricQuery {
    /// Create a new ElectricQuery from a tree-sitter query source string.
    pub fn new(
        grammar: Grammar,
        source: &str,
    ) -> Result<Self, crate::tree_sitter::query::ParseError> {
        let query = Query::new(grammar, source, |_pattern, _predicate| Ok(()))?;
        let electric_keyword = query.get_capture("electric-keyword");

        Ok(Self {
            query,
            electric_keyword,
        })
    }

    /// Whether the line containing `pos` should be outdented right now.
    ///
    /// This is the case when the line starts with an `@electric-keyword`
    /// token, the cursor is at the end of the line, and at least one further
    /// character has been typed after the keyword.
    ///
    /// Requiring the keyword to be the first token on the line keeps
    /// expressions like `x = 1 if c else 2` from triggering an outdent.
    /// Requiring a character after it is what distinguishes `else:` from an
    /// identifier that merely starts with a keyword: at the moment `else` is
    /// complete `elsewhere` is indistinguishable from `else`, but one
    /// character later the grammar has lexed it as a single identifier and no
    /// keyword matches any more.
    pub fn should_outdent(&self, syntax: &Syntax, text: RopeSlice, pos: usize) -> bool {
        let Some(electric_keyword) = self.electric_keyword else {
            return false;
        };

        let line_idx = text.char_to_line(pos);
        let line_start = text.line_to_char(line_idx);
        let Some(first_non_whitespace) = text.line(line_idx).first_non_whitespace_char() else {
            return false;
        };
        // No indentation to remove.
        if first_non_whitespace == 0 {
            return false;
        }
        let token_start = line_start + first_non_whitespace;
        if token_start >= pos {
            return false;
        }
        // The keyword must be the last thing on the line: anything after the
        // cursor means we are editing an existing line rather than typing a
        // fresh clause.
        if text
            .slice(pos..)
            .line(0)
            .chars()
            .any(|c| !c.is_whitespace())
        {
            return false;
        }

        let start_byte = text.char_to_byte(token_start) as u32;
        let end_byte = text.char_to_byte(pos) as u32;

        let tree = syntax.tree();
        let root = tree.root_node();
        let mut cursor = InactiveQueryCursor::new(start_byte..end_byte, TREE_SITTER_MATCH_LIMIT)
            .execute_query(&self.query, &root, RopeInput::new(text));

        while let Some(mat) = cursor.next_match() {
            for matched_node in mat.matched_nodes() {
                if matched_node.capture == electric_keyword
                    && matched_node.node.start_byte() == start_byte
                    && matched_node.node.end_byte() < end_byte
                {
                    return true;
                }
            }
        }

        false
    }
}
