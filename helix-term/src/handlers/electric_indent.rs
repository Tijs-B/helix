use helix_core::{
    indent::indent_level_for_line, syntax::config::IndentationHeuristic, Tendril, Transaction,
};
use helix_event::register_hook;
use helix_stdx::rope::RopeSliceExt;

use crate::events::PostInsertChar;
use crate::handlers::Handlers;

pub(super) fn register_hooks(_handlers: &Handlers) {
    register_hook!(move |event: &mut PostInsertChar<'_, '_>| {
        electric_outdent(event);
        Ok(())
    });
}

/// Outdent the current line by one level when an "electric" keyword such as
/// python's `else` has just been completed.
///
/// The tree-sitter indent query cannot be used to compute the new indentation
/// here: for indentation sensitive grammars the keyword only parses as part of
/// its parent statement *after* it has been outdented, so the tree we would
/// have to ask is the one we are trying to produce. Removing a single indent
/// level is what the keyword means in every language this applies to.
fn electric_outdent(event: &mut PostInsertChar) {
    let editor = &mut event.cx.editor;

    if !matches!(
        editor.config().indent_heuristic,
        IndentationHeuristic::TreeSitter | IndentationHeuristic::Hybrid
    ) {
        return;
    }

    let (view, doc) = current_ref!(editor);
    let view_id = view.id;
    let Some(syntax) = doc.syntax() else {
        return;
    };

    let loader = editor.syn_loader.load();
    let Some(electric_query) = loader.electric_query(syntax.root_language()) else {
        return;
    };

    let text = doc.text().slice(..);
    let selection = doc.selection(view_id);

    // Only a single cursor is handled: with multiple cursors on different
    // lines the keyword has generally only been completed at some of them.
    if selection.len() != 1 {
        return;
    }
    let pos = selection.primary().cursor(text);

    if !electric_query.should_outdent(syntax, text, pos) {
        return;
    }

    let line_idx = text.char_to_line(pos);
    let line_start = text.line_to_char(line_idx);
    let Some(first_non_whitespace) = text.line(line_idx).first_non_whitespace_char() else {
        return;
    };

    let indent_style = doc.indent_style;
    let tab_width = doc.tab_width();
    let indent_width = indent_style.indent_width(tab_width);

    // Outdent one level relative to the *previous* non-blank line rather than
    // relative to the current one. Anchoring to a line we don't modify makes
    // this idempotent, so a keyword can never be outdented twice, and means a
    // keyword that is already at or left of its target is left alone.
    let Some(previous) = (0..line_idx)
        .rev()
        .map(|line| text.line(line))
        .find(|line| line.first_non_whitespace_char().is_some())
    else {
        return;
    };
    let new_level = indent_level_for_line(previous, tab_width, indent_width).saturating_sub(1);
    if new_level >= indent_level_for_line(text.line(line_idx), tab_width, indent_width) {
        return;
    }
    let new_indent = indent_style.as_str().repeat(new_level);

    let (view, doc) = current!(editor);
    let transaction = Transaction::change(
        doc.text(),
        [(
            line_start,
            line_start + first_non_whitespace,
            (!new_indent.is_empty()).then(|| Tendril::from(new_indent)),
        )]
        .into_iter(),
    );
    doc.apply(&transaction, view.id);
}
