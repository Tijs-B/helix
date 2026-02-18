//! When typing the opening character of one of the possible pairs defined below,
//! this module provides the functionality to insert the paired closing character.

use crate::{graphemes, movement::Direction, Range, Rope, Selection, Tendril, Transaction};
use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

// Heavily based on https://github.com/codemirror/closebrackets/
pub const DEFAULT_PAIRS: &[(&str, &str)] = &[
    ("(", ")"),
    ("{", "}"),
    ("[", "]"),
    ("'", "'"),
    ("\"", "\""),
    ("`", "`"),
];

/// Represents the config for a particular pairing.
#[derive(Debug, Clone)]
pub struct Pair {
    pub open: String,
    pub close: String,
}

impl Pair {
    /// true if open == close
    pub fn same(&self) -> bool {
        self.open == self.close
    }

    /// true if both open and close are single characters
    pub fn is_single_char(&self) -> bool {
        self.open.len() == 1 && self.close.len() == 1
    }

    /// true if all of the pair's conditions hold for the given document and range
    pub fn should_close(&self, doc: &Rope, range: &Range) -> bool {
        let mut should_close = next_is_not_alpha(doc, range);

        // Only check prev_is_not_alpha for single-char same-pairs (e.g. ', ", `)
        // Multi-char openers explicitly include their preceding context
        if self.same() && self.is_single_char() {
            should_close &= prev_is_not_alpha(doc, range);
        }

        should_close
    }
}

fn next_is_not_alpha(doc: &Rope, range: &Range) -> bool {
    let cursor = range.cursor(doc.slice(..));
    doc.get_char(cursor)
        .map(|c| !c.is_alphanumeric())
        .unwrap_or(true)
}

fn prev_is_not_alpha(doc: &Rope, range: &Range) -> bool {
    let cursor = range.cursor(doc.slice(..));
    prev_char(doc, cursor)
        .map(|c| !c.is_alphanumeric())
        .unwrap_or(true)
}

/// The type that represents the collection of auto pairs.
#[derive(Debug, Clone)]
pub struct AutoPairs {
    pairs: Vec<Pair>,
    /// Maps the last char of an opener to pair indices (longest opener first)
    opener_index: HashMap<char, Vec<usize>>,
    /// Set of chars that are the first char of any closer
    closer_chars: HashSet<char>,
}

impl AutoPairs {
    pub fn new(pairs: impl IntoIterator<Item = Pair>) -> Self {
        let pairs: Vec<Pair> = pairs.into_iter().collect();

        let mut opener_index: HashMap<char, Vec<usize>> = HashMap::new();
        let mut closer_chars = HashSet::new();

        for (idx, pair) in pairs.iter().enumerate() {
            if let Some(last_char) = pair.open.chars().last() {
                opener_index.entry(last_char).or_default().push(idx);
            }
            if let Some(first_char) = pair.close.chars().next() {
                closer_chars.insert(first_char);
            }
        }

        // Sort each opener list by opener length (longest first) for greedy matching
        for indices in opener_index.values_mut() {
            indices.sort_by(|a, b| pairs[*b].open.len().cmp(&pairs[*a].open.len()));
        }

        Self {
            pairs,
            opener_index,
            closer_chars,
        }
    }

    /// Find the longest opener ending with `ch` that matches the preceding text.
    pub fn find_opener(&self, doc: &Rope, range: &Range, ch: char) -> Option<&Pair> {
        let indices = self.opener_index.get(&ch)?;
        let cursor = range.cursor(doc.slice(..));

        for &idx in indices {
            let pair = &self.pairs[idx];
            let prefix_len = pair.open.len() - ch.len_utf8();

            if prefix_len == 0 {
                return Some(pair);
            }

            // Check if the text before the cursor matches the prefix
            if cursor >= prefix_len {
                let start = cursor - prefix_len;
                let doc_prefix = doc.slice(start..cursor);
                let opener_prefix = &pair.open[..prefix_len];
                if doc_prefix.len_chars() == opener_prefix.chars().count()
                    && doc_prefix.chars().eq(opener_prefix.chars())
                {
                    return Some(pair);
                }
            }
        }

        None
    }

    /// Check if `ch` is the first char of any closer.
    pub fn is_closer_char(&self, ch: char) -> bool {
        self.closer_chars.contains(&ch)
    }

    /// Find the closer that a shorter pair would have auto-inserted for the
    /// prefix of a multi-char opener. For example, if `{%` is an opener and
    /// `{` is also an opener (with closer `}`), this returns `}` — the closer
    /// that was auto-inserted when `{` was typed and now needs to be removed.
    pub fn find_superseded_closer(&self, pair: &Pair) -> Option<&str> {
        if pair.open.len() <= 1 {
            return None;
        }
        // The prefix is everything except the last char of the opener.
        // Check if that prefix itself is an opener for a shorter pair.
        let prefix = &pair.open[..pair.open.len() - pair.open.chars().last().unwrap().len_utf8()];
        self.pairs
            .iter()
            .find(|p| p.open == prefix && !p.same())
            .map(|p| p.close.as_str())
    }

    /// Compatibility method: get a single-char pair by open or close char.
    /// Used by commands.rs for enter-between-pairs and backspace logic.
    pub fn get(&self, ch: char) -> Option<&Pair> {
        self.pairs.iter().find(|pair| {
            pair.is_single_char()
                && (pair.open.starts_with(ch) || pair.close.starts_with(ch))
        })
    }
}

impl Default for AutoPairs {
    fn default() -> Self {
        AutoPairs::new(DEFAULT_PAIRS.iter().map(|&(open, close)| Pair {
            open: open.into(),
            close: close.into(),
        }))
    }
}

#[must_use]
pub fn hook(doc: &Rope, selection: &Selection, ch: char, pairs: &AutoPairs) -> Option<Transaction> {
    log::trace!("autopairs hook selection: {:#?}", selection);

    // 1. Find longest opener ending with ch whose prefix matches preceding text
    if let Some(pair) = pairs.find_opener(doc, &selection.primary(), ch) {
        if pair.same() {
            return Some(handle_same(doc, selection, pair, ch));
        }
        return Some(handle_open(doc, selection, pair, ch, pairs));
    }

    // 2. Check if ch matches a closer (skip over)
    if pairs.is_closer_char(ch) {
        return Some(handle_close(doc, selection, ch));
    }

    None
}

fn prev_char(doc: &Rope, pos: usize) -> Option<char> {
    if pos == 0 {
        return None;
    }

    doc.get_char(pos - 1)
}

/// Calculate what the resulting range should be for an auto pair insertion.
///
/// `len_inserted` is the total number of characters inserted (used for offset
/// accumulation across multiple ranges). `cursor_offset` is how far the cursor
/// should advance from the insertion point (always 1 for pair insertions, 0 for
/// skip-over).
fn get_next_range(
    doc: &Rope,
    start_range: &Range,
    offset: usize,
    len_inserted: usize,
    cursor_offset: usize,
) -> Range {
    // inserting at the very end of the document after the last newline
    if start_range.head == doc.len_chars() && start_range.anchor == doc.len_chars() {
        return Range::new(
            start_range.anchor + offset + cursor_offset,
            start_range.head + offset + cursor_offset,
        );
    }

    let doc_slice = doc.slice(..);
    let single_grapheme = start_range.is_single_grapheme(doc_slice);

    // just skip over graphemes
    if len_inserted == 0 {
        let end_anchor = if single_grapheme {
            graphemes::next_grapheme_boundary(doc_slice, start_range.anchor) + offset
        } else {
            start_range.anchor + offset
        };

        return Range::new(
            end_anchor,
            graphemes::next_grapheme_boundary(doc_slice, start_range.head) + offset,
        );
    }

    // trivial case: only inserted a single char (no closer), just move the selection
    if len_inserted == 1 {
        let end_anchor = if single_grapheme || start_range.direction() == Direction::Backward {
            start_range.anchor + offset + 1
        } else {
            start_range.anchor + offset
        };

        return Range::new(end_anchor, start_range.head + offset + 1);
    }

    // For pair insertions (len_inserted >= 2), cursor_offset + 1 represents
    // how far the head should advance. This equals len_inserted for single-char
    // pairs (2 = 1+1) but differs for multi-char closers (e.g. 2 vs 4 for """).
    let head_advance = cursor_offset + 1;

    let end_head = if start_range.head == 0 || start_range.direction() == Direction::Backward {
        start_range.head + offset + cursor_offset
    } else {
        let prev_bound = graphemes::prev_grapheme_boundary(doc_slice, start_range.head);
        log::trace!(
            "prev_bound: {}, offset: {}, head_advance: {}",
            prev_bound,
            offset,
            head_advance
        );
        prev_bound + offset + head_advance
    };

    let end_anchor = match (start_range.len(), start_range.direction()) {
        (0, _) => end_head,

        (1, Direction::Forward) => end_head - 1,
        (1, Direction::Backward) => end_head + 1,

        (_, Direction::Forward) => {
            if single_grapheme {
                graphemes::prev_grapheme_boundary(doc.slice(..), start_range.head) + 1
            } else {
                start_range.anchor + offset
            }
        }

        (_, Direction::Backward) => {
            if single_grapheme {
                graphemes::prev_grapheme_boundary(doc.slice(..), start_range.anchor)
                    + head_advance
                    + offset
            } else {
                start_range.anchor + offset + len_inserted
            }
        }
    };

    Range::new(end_anchor, end_head)
}

/// Produce the change and range for a single cursor in a pair-insertion transaction.
fn make_pair_change(
    doc: &Rope,
    start_range: &Range,
    offs: &mut usize,
    end_ranges: &mut SmallVec<[Range; 1]>,
    len_inserted: usize,
) {
    let cursor_offset = std::cmp::min(len_inserted, 1);
    let next_range = get_next_range(doc, start_range, *offs, len_inserted, cursor_offset);
    end_ranges.push(next_range);
    *offs += len_inserted;
}

fn handle_open(
    doc: &Rope,
    selection: &Selection,
    pair: &Pair,
    ch: char,
    pairs: &AutoPairs,
) -> Transaction {
    let mut end_ranges = SmallVec::with_capacity(selection.len());
    let mut offs = 0;

    // Check if a shorter pair's closer was auto-inserted and needs removal.
    // For example, typing '{' auto-inserts '}', then typing '%' to complete
    // the '{%' opener should replace that '}' with the correct closer '%}'.
    let superseded_closer = pairs.find_superseded_closer(pair);

    let transaction = Transaction::change_by_selection(doc, selection, |start_range| {
        let cursor = start_range.cursor(doc.slice(..));

        let (len_inserted, change) = if !pair.should_close(doc, start_range) {
            let mut tendril = Tendril::new();
            tendril.push(ch);
            (1, (cursor, cursor, Some(tendril)))
        } else {
            let mut pair_str = Tendril::new();
            pair_str.push(ch);
            pair_str.push_str(&pair.close);

            // If a shorter pair's closer sits right at the cursor, delete it
            let delete_len = superseded_closer
                .filter(|closer| {
                    let close_chars = closer.chars().count();
                    let available = doc.len_chars() - cursor;
                    available >= close_chars
                        && doc
                            .slice(cursor..cursor + close_chars)
                            .chars()
                            .eq(closer.chars())
                })
                .map(|closer| closer.chars().count())
                .unwrap_or(0);

            let len = 1 + pair.close.chars().count();
            (len, (cursor, cursor + delete_len, Some(pair_str)))
        };

        make_pair_change(doc, start_range, &mut offs, &mut end_ranges, len_inserted);
        change
    });

    let t = transaction.with_selection(Selection::new(end_ranges, selection.primary_index()));
    log::debug!("auto pair transaction: {:#?}", t);
    t
}

fn handle_close(doc: &Rope, selection: &Selection, ch: char) -> Transaction {
    let mut end_ranges = SmallVec::with_capacity(selection.len());
    let mut offs = 0;

    let transaction = Transaction::change_by_selection(doc, selection, |start_range| {
        let cursor = start_range.cursor(doc.slice(..));

        let (len_inserted, change) = if doc.get_char(cursor) == Some(ch) {
            (0, (cursor, cursor, None))
        } else {
            let mut tendril = Tendril::new();
            tendril.push(ch);
            (1, (cursor, cursor, Some(tendril)))
        };

        make_pair_change(doc, start_range, &mut offs, &mut end_ranges, len_inserted);
        change
    });

    let t = transaction.with_selection(Selection::new(end_ranges, selection.primary_index()));
    log::debug!("auto pair transaction: {:#?}", t);
    t
}

/// Handle cases where open and close is the same, or in triples ("""docstring""")
fn handle_same(doc: &Rope, selection: &Selection, pair: &Pair, ch: char) -> Transaction {
    let mut end_ranges = SmallVec::with_capacity(selection.len());
    let mut offs = 0;

    let transaction = Transaction::change_by_selection(doc, selection, |start_range| {
        let cursor = start_range.cursor(doc.slice(..));

        // Check if text at cursor starts with the closer
        let next_matches_closer = {
            let close_len = pair.close.chars().count();
            let available = doc.len_chars() - cursor;
            available >= close_len
                && doc.slice(cursor..cursor + close_len).chars().eq(pair.close.chars())
        };

        let (len_inserted, change) = if next_matches_closer {
            (0, (cursor, cursor, None))
        } else {
            let mut pair_str = Tendril::new();
            pair_str.push(ch);

            if pair.should_close(doc, start_range) {
                pair_str.push_str(&pair.close);
            }

            let len = pair_str.chars().count();
            (len, (cursor, cursor, Some(pair_str)))
        };

        make_pair_change(doc, start_range, &mut offs, &mut end_ranges, len_inserted);
        change
    });

    let t = transaction.with_selection(Selection::new(end_ranges, selection.primary_index()));
    log::debug!("auto pair transaction: {:#?}", t);
    t
}
