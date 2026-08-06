use smartstring::{LazyCompact, SmartString};
use textwrap::{Options, WordSplitter::NoHyphenation};

/// Given a slice of text, return the text re-wrapped to fit it
/// within the given width.
pub fn reflow_hard_wrap(text: &str, text_width: usize) -> SmartString<LazyCompact> {
    let (unfilled, mut indent) = textwrap::unfill(text);

    // `unfill` only infers a subsequent-line indent (e.g. leading
    // whitespace plus a comment token like `// ` or `# `) by comparing
    // at least two physical lines. A single overflowing line has
    // nothing to compare against, so fall back to reusing the first
    // line's indent for the lines produced by wrapping, keeping
    // continuation lines commented and indented the same way.
    if indent.subsequent_indent.is_empty() {
        indent.subsequent_indent = indent.initial_indent;
    }

    let options = Options::new(text_width)
        .initial_indent(indent.initial_indent)
        .subsequent_indent(indent.subsequent_indent)
        .word_splitter(NoHyphenation)
        .word_separator(textwrap::WordSeparator::AsciiSpace);

    let stripped = unfilled
        .strip_suffix(indent.line_ending.as_str())
        .unwrap_or(&unfilled);
    let mut refilled: SmartString<LazyCompact> = textwrap::fill(stripped, options).into();
    if stripped.len() != unfilled.len() {
        refilled.push_str(indent.line_ending.as_str());
    }
    refilled
}

#[cfg(test)]
mod test {
    use super::reflow_hard_wrap;

    #[test]
    fn reflow_single_line_comment_preserves_prefix_on_continuation_lines() {
        let text = "    # A very long paragraph with a lot of words, overflowing the max text width, and even some more text. Lorem ipsum dolor sit amet blah blah blah";
        let reflowed = reflow_hard_wrap(text, 80);
        for line in reflowed.lines() {
            assert!(line.starts_with("    # "), "line missing prefix: {line:?}");
        }
    }

    #[test]
    fn reflow_plain_single_line_has_no_indent() {
        let text = "A very long paragraph with a lot of words, overflowing the max text width, and even some more text.";
        let reflowed = reflow_hard_wrap(text, 40);
        assert!(reflowed.lines().count() > 1);
        for line in reflowed.lines() {
            assert!(!line.starts_with(' '));
        }
    }
}
