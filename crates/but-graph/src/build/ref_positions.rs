//! The editor-grade ref layout: every surfaced reference in row order with its position over
//! the commit graph, plus the entrypoint's reach (mutability), HEAD ordinals, and the
//! workspace commit's chain slots.
//!
//! Derived once per build from the [`SegmentData`] and stored on the commit graph's
//! [`RefLayout`](crate::ref_layout::RefLayout); the rebase editor is a pure consumer of
//! this layout.

use std::collections::HashMap;

use anyhow::{Context as _, Result};

use super::IdMap;
use super::segment_data::{OrderedTargets, SegmentData};
use crate::ref_layout::{PositionedRef, RefPosition, RefPositions};

/// One entry in a row's linearization.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Index into `ref_table`.
    Ref(usize),
    /// Index into `commits`.
    Commit(usize),
    /// The placeholder for a row with neither name nor commits.
    None,
}

/// Derive the stored layout from the segments in `data`. The managed-workspace decision comes from
/// `cg`, which recognises managed commits by message.
#[tracing::instrument(level = "trace", skip_all)]
pub(super) fn derive_ref_positions(
    data: &SegmentData,
    cg: &crate::CommitGraph,
) -> Result<RefPositions> {
    let entrypoint_sidx = data
        .entrypoint_sidx
        .context("BUG: must always set the entrypoint")?;
    let workspace_commit_id = data.entrypoint.filter(|&id| cg.is_managed_ws_commit(id));
    Ok(positions_from_segments(
        &data.segments,
        &data.parent_ordered_targets(),
        entrypoint_sidx,
        workspace_commit_id,
    ))
}

/// Compute the layout from the segments: per-segment step rows (segment ref, then per commit its refs then the
/// commit), the parent fixup (a commit whose group-flattened parents disagree with its raw
/// parent list is rewired directly, bypassing groups — the ws commit and partially-traversed
/// commits keep their wiring), position derivation, and the strip's slot compaction.
///
/// `workspace_commit_id` is the managed entrypoint commit, which takes its resolved CHAIN
/// slots instead of its real parents.
fn positions_from_segments(
    segments: &[crate::Segment],
    targets_of_segment: &OrderedTargets,
    entrypoint_sidx: usize,
    workspace_commit_id: Option<gix::ObjectId>,
) -> RefPositions {
    // Reach: every segment the entrypoint's segment descends into through the parent-ordered edges.
    let mut reachable_sidx_at = vec![false; segments.len()];
    let mut queue = vec![entrypoint_sidx];
    while let Some(ri) = queue.pop() {
        if !std::mem::replace(&mut reachable_sidx_at[ri], true) {
            queue.extend(targets_of_segment.of(ri));
        }
    }
    let head_name = segments[entrypoint_sidx].ref_name().map(|n| n.to_owned());

    // Step build: one row of steps per segment, in table order.
    let mut steps: Vec<Step> = Vec::new();
    let mut parents: Vec<Vec<usize>> = Vec::new();
    let mut ref_table: Vec<(gix::refs::FullName, bool)> = Vec::new();
    let mut commits: Vec<(gix::ObjectId, &[gix::ObjectId])> = Vec::new();
    // Commit `c`'s step handle, positionally — the id-keyed map only serves parent-id lookups.
    let mut commit_step_at: Vec<usize> = Vec::new();
    let mut commit_step = IdMap::<usize>::default();
    let mut reachable_commits = Vec::new();
    let mut head_refs = Vec::new();
    let mut step_rows = Vec::new();

    for (ri, data) in segments.iter().enumerate() {
        let reachable = reachable_sidx_at[ri];
        let mut row: Vec<usize> = vec![];
        let push = |steps: &mut Vec<Step>, parents: &mut Vec<Vec<usize>>, step| {
            steps.push(step);
            parents.push(vec![]);
            steps.len() - 1
        };

        if let Some(reference) = data.ref_name() {
            if head_name.as_ref().map(|n| n.as_ref()) == Some(reference) {
                head_refs.push(ref_table.len());
            }
            ref_table.push((reference.to_owned(), reachable));
            let n = push(&mut steps, &mut parents, Step::Ref(ref_table.len() - 1));
            row.push(n);
        }
        for commit in &data.commits {
            if reachable {
                reachable_commits.push(commit.id);
            }
            for r in &commit.refs {
                ref_table.push((r.ref_name.clone(), reachable));
                let n = push(&mut steps, &mut parents, Step::Ref(ref_table.len() - 1));
                if let Some(&previous) = row.last() {
                    parents[previous].push(n);
                }
                row.push(n);
            }
            commits.push((commit.id, commit.parent_ids.as_slice()));
            let n = push(&mut steps, &mut parents, Step::Commit(commits.len() - 1));
            commit_step_at.push(n);
            commit_step.insert(commit.id, n);
            if let Some(&previous) = row.last() {
                parents[previous].push(n);
            }
            row.push(n);
        }
        if row.is_empty() {
            row.push(push(&mut steps, &mut parents, Step::None));
        }
        step_rows.push(row);
    }

    // The parent-ordered edges land on each row's LAST step, pointing at the target row's FIRST step.
    let first_step_of_row: Vec<usize> = step_rows
        .iter()
        .map(|row| *row.first().expect("every row has a step"))
        .collect();
    for (ri, row) in step_rows.iter().enumerate() {
        let source = *row.last().expect("every row has a step");
        for &target in targets_of_segment.of(ri) {
            parents[source].push(first_step_of_row[target]);
        }
    }

    // The fixup: flatten a commit's group parents in slot order; on disagreement with the
    // RAW parent list, rewire directly to present commits (groups lose their edges). The ws
    // commit and partially-traversed commits keep their segment wiring. The flattening
    // compares in-place against the raw list — no per-commit id collection.
    let mut stack: Vec<usize> = Vec::new();
    let flatten_matches = |steps: &[Step],
                           parents: &[Vec<usize>],
                           commits: &[(gix::ObjectId, &[gix::ObjectId])],
                           start: usize,
                           want: &[gix::ObjectId],
                           stack: &mut Vec<usize>|
     -> bool {
        stack.clear();
        stack.extend(parents[start].iter().rev().copied());
        let mut matched = 0;
        while let Some(n) = stack.pop() {
            match steps[n] {
                Step::Commit(c) => {
                    if want.get(matched) != Some(&commits[c].0) {
                        return false;
                    }
                    matched += 1;
                }
                Step::Ref(_) | Step::None => {
                    stack.extend(parents[n].iter().rev().copied());
                }
            }
        }
        matched == want.len()
    };
    for (ci, &(id, raw_parents)) in commits.iter().enumerate() {
        if Some(id) == workspace_commit_id {
            continue;
        }
        let preserved =
            !raw_parents.is_empty() && raw_parents.iter().any(|p| !commit_step.contains_key(p));
        if preserved {
            continue;
        }
        let n = commit_step_at[ci];
        if flatten_matches(&steps, &parents, &commits, n, raw_parents, &mut stack) {
            continue;
        }
        parents[n] = raw_parents
            .iter()
            .filter_map(|p| commit_step.get(p).copied())
            .collect();
    }

    // Positions from the (post-fixup, pre-strip) topology: descend first-edges for `on` and
    // below, ascend for the entering edges and the convergence signal. The reverse adjacency
    // is CSR-shaped (offsets into one flat run) — a Vec per step costs an allocation per commit.
    let mut incoming_start = vec![0usize; steps.len() + 1];
    for slots in &parents {
        for &parent in slots {
            incoming_start[parent + 1] += 1;
        }
    }
    for i in 0..steps.len() {
        incoming_start[i + 1] += incoming_start[i];
    }
    let mut incoming_flat: Vec<(usize, usize)> =
        vec![(0, 0); *incoming_start.last().expect("n+1 offsets")];
    {
        let mut fill = incoming_start.clone();
        for (child, slots) in parents.iter().enumerate() {
            for (slot, &parent) in slots.iter().enumerate() {
                incoming_flat[fill[parent]] = (child, slot);
                fill[parent] += 1;
            }
        }
    }
    let incoming = |n: usize| &incoming_flat[incoming_start[n]..incoming_start[n + 1]];
    let is_commit = |n: usize| matches!(steps[n], Step::Commit(_));
    let ref_steps: Vec<(usize, usize)> = steps
        .iter()
        .enumerate()
        .filter_map(|(n, step)| match step {
            Step::Ref(r) => Some((n, *r)),
            _ => None,
        })
        .collect();
    struct DerivedPosition {
        on: usize,
        below: Option<usize>,
        ambiguous: bool,
        entering: Vec<(usize, usize)>,
    }
    let mut positions = HashMap::<usize, DerivedPosition>::new();
    for &(ref_step, _) in &ref_steps {
        let mut cursor = ref_step;
        let mut on = None;
        let mut below = None;
        for _ in 0..10_000 {
            let Some(&next) = parents[cursor].first() else {
                break;
            };
            if is_commit(next) {
                on = Some(next);
                break;
            }
            if matches!(steps[next], Step::Ref(_)) && below.is_none() {
                below = Some(next);
            }
            cursor = next;
        }
        let Some(on) = on else {
            continue; // unborn: no stored position
        };
        let mut cursor = ref_step;
        let mut entering_edges = Vec::new();
        let mut ambiguous = false;
        for _ in 0..10_000 {
            let entering = incoming(cursor);
            let picks: Vec<_> = entering
                .iter()
                .copied()
                .filter(|&(child, _)| is_commit(child))
                .collect();
            if !picks.is_empty() {
                ambiguous = entering.len() > 1;
                entering_edges = picks;
                break;
            }
            let mut others = entering.iter().filter(|&&(child, _)| !is_commit(child));
            match (others.next(), others.next()) {
                (Some(&(child, _)), None) => cursor = child,
                _ => break,
            }
        }
        positions.insert(
            ref_step,
            DerivedPosition {
                on,
                below,
                ambiguous,
                entering: entering_edges,
            },
        );
    }

    // The strip's slot compaction: resolve each commit's parent entries to commits (dropping
    // unborn groups), record the vacated slots so the captured entering edges can be renamed,
    // and keep the ws commit's resolved CHAIN slots (one per chain, so empty chains
    // over one base yield duplicate entries the real commit does not have).
    let resolve = |start: usize| -> Option<usize> {
        let mut cursor = start;
        for _ in 0..10_000 {
            if is_commit(cursor) {
                return Some(cursor);
            }
            cursor = *parents[cursor].first()?;
        }
        None
    };
    let mut dropped: Vec<(usize, usize)> = Vec::new();
    let mut workspace_chain_slots: Option<(gix::ObjectId, Vec<gix::ObjectId>)> = None;
    for (ci, &(id, _)) in commits.iter().enumerate() {
        let n = commit_step_at[ci];
        // Only the ws commit keeps its resolved ids — everyone else just detects vacated slots.
        let is_ws = Some(id) == workspace_commit_id;
        let mut resolved = Vec::new();
        for (slot, &parent) in parents[n].iter().enumerate() {
            match resolve(parent) {
                Some(pick) if is_ws => {
                    let Step::Commit(c) = steps[pick] else {
                        unreachable!("resolve returns commits");
                    };
                    resolved.push(commits[c].0);
                }
                Some(_) => {}
                None => dropped.push((n, slot)),
            }
        }
        if is_ws {
            workspace_chain_slots = Some((id, resolved));
        }
    }
    for position in positions.values_mut() {
        for (edge_source, slot) in position.entering.iter_mut() {
            *slot -= dropped
                .iter()
                .filter(|(source, vacated)| source == edge_source && vacated < slot)
                .count();
        }
    }

    // The stored shape: ref-table order, step handles translated to ref ordinals and commit
    // ids, entering edges sorted.
    let mut step_of_ref = vec![0usize; ref_table.len()];
    for &(n, r) in &ref_steps {
        step_of_ref[r] = n;
    }
    let refs = ref_table
        .into_iter()
        .enumerate()
        .map(|(r, (name, reachable))| {
            let position = positions.get(&step_of_ref[r]).map(|position| {
                let Step::Commit(c) = steps[position.on] else {
                    unreachable!("positions sit on commits");
                };
                let below = position.below.map(|b| {
                    let Step::Ref(br) = steps[b] else {
                        unreachable!("below entries are refs");
                    };
                    br
                });
                let mut entering: Vec<(gix::ObjectId, usize)> = position
                    .entering
                    .iter()
                    .map(|&(child, slot)| {
                        let Step::Commit(c) = steps[child] else {
                            unreachable!("entering edges come from commits");
                        };
                        (commits[c].0, slot)
                    })
                    .collect();
                entering.sort_unstable();
                RefPosition {
                    on: commits[c].0,
                    below,
                    entering,
                    ambiguous: position.ambiguous,
                }
            });
            PositionedRef {
                name,
                reachable,
                position,
            }
        })
        .collect();

    reachable_commits.sort_unstable();
    RefPositions {
        refs,
        workspace_chain_slots,
        head_refs,
        reachable_commits,
    }
}
