//! Phase 3 of gather-then-build: author the segment data from the commit-graph and the
//! chain plan, then render the segment graph from it.

use std::collections::{HashMap, HashSet};

use gix::reference::Category;

use super::IdSet;
use super::chains::{advanced_outside_decisions, empty_workspace_ref};
use super::facts::Facts;
use super::plan::ChainPlan;
use super::remotes::remote_name_in_play;
use crate::{Commit, CommitGraph, RefInfo};

/// Materialize the [`SegmentData`](super::segment_data::SegmentData) — the record — from
/// `b.cg`, plus the [`Graph`](crate::Graph) shell it will render into.
///
/// `b` is the build context every phase reads; `b.project_meta`/`b.options` are carried onto
/// the `Graph`.
///
/// This is "gather-then-build": everything is decided as data BEFORE any segment exists —
/// the caller authors `f`/`plan`/`layout` via `gather_and_plan`; this function adds the
/// repository/metadata-flavored decisions (the empty-workspace splice name, the advanced-outside
/// branches, the claimed remote names), then [`build`](super::segment_data::build)
/// materializes the record. The returned graph has NO segments yet — after any ad-hoc
/// rewrites and the ref-position derivation, the assembler's final
/// [`render_segment_graph`](super::segment_data::render_segment_graph) consumes the record,
/// installing its segments as the graph's storage (position == segment id).
#[tracing::instrument(
    name = "graph_from_commit_graph",
    level = "trace",
    skip_all,
    fields(commits = b.cg.node_count())
)]
pub(crate) fn graph_from_commit_graph<T: but_core::RefMetadata>(
    b: &super::BuildInputs<'_>,
    meta: &T,
    f: Facts,
    plan: &ChainPlan,
    layout: &crate::ref_layout::RefLayout,
) -> (crate::Graph, super::segment_data::SegmentData) {
    let &super::BuildInputs {
        cg,
        workspace_commit,
        entrypoint,
        entrypoint_ref,
        remote_tracking,
        symbolic_remotes,
        stack_branches,
        managed,
        project_meta,
        options,
        ..
    } = b;
    let Facts {
        in_set,
        ws_is_managed_merge: _,
        empty_ws_case,
        pinned_commits,
        boundaries,
        entrypoint_forced_boundary: _,
        owner_of,
        tips,
    } = f;

    // Chain-structure decisions: the empty-workspace splice's name and the advanced-outside
    // branches.
    let ws_empty_ref = (managed && empty_ws_case)
        .then(|| empty_workspace_ref(cg, workspace_commit))
        .flatten();
    let advanced_outside = if managed {
        let mut named: HashSet<gix::refs::FullName> = tips
            .iter()
            .filter_map(|&tip| plan.tip_name(tip).map(|(name, _)| name))
            .collect();
        named.extend(plan.floats.iter().map(|fl| fl.name.clone()));
        named.extend(ws_empty_ref.clone());
        advanced_outside_decisions(
            cg,
            &in_set,
            &owner_of,
            stack_branches,
            remote_tracking,
            meta,
            project_meta.target_ref.as_ref(),
            &pinned_commits,
            named,
        )
    } else {
        Vec::new()
    };

    let claimed_remote_names = claim_remote_names(
        cg,
        plan,
        &in_set,
        remote_tracking,
        stack_branches,
        symbolic_remotes,
    );
    // The entrypoint is a planned boundary in every region too: a checkout inside a remote's
    // ahead run starts its own segment at creation, never split out after the fact.
    let region_pinned = {
        let mut p = pinned_commits.clone();
        p.insert(entrypoint);
        p
    };
    let segment_data = super::segment_data::build(super::segment_data::GraphInputs {
        cg,
        tips: &tips,
        in_set: &in_set,
        boundaries: &boundaries,
        owner_of: &owner_of,
        plan,
        layout,
        workspace_commit,
        managed,
        ws_empty_ref: ws_empty_ref.as_ref(),
        advanced_outside: &advanced_outside,
        remote_tracking,
        symbolic_remotes,
        stack_branches,
        region_pinned: &region_pinned,
        claimed_remote_names: &claimed_remote_names,
        entrypoint,
        entrypoint_ref,
        target_ref: project_meta.target_ref.as_ref(),
        extra_target: options.extra_target_commit_id,
    });
    let graph = crate::Graph {
        entrypoint: segment_data.graph_entrypoint(),
        entrypoint_ref: entrypoint_ref.cloned(),
        project_meta: project_meta.clone(),
        options: options.clone(),
        ..crate::Graph::default()
    };
    (graph, segment_data)
}

/// Remote refs some creator will consume as a segment name: the region builder cuts its run
/// at interior remote refs only when unclaimed. Plan-modeled names (`remote_used` covers the
/// walk seeds) plus the ahead-case remotes of EVERY boundary-tip local (`add_remote_segments`
/// regions all of them, mirroring its gates) plus explicit-tip remote names.
fn claim_remote_names(
    cg: &CommitGraph,
    plan: &ChainPlan,
    in_set: &IdSet,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
    stack_branches: Option<&[Vec<gix::refs::FullName>]>,
    symbolic_remotes: &[String],
) -> HashSet<gix::refs::FullName> {
    let mut claimed_remote_names: HashSet<gix::refs::FullName> = plan.remote_used.clone();
    claimed_remote_names.extend(plan.base_name_of.values().filter_map(|name| {
        let rt = remote_tracking.get(name)?;
        let rt_tip = cg.commit_by_ref(rt.as_ref())?;
        let is_meta_stack_branch = stack_branches
            .into_iter()
            .flatten()
            .flatten()
            .any(|b| b == rt);
        (!in_set.contains(&rt_tip)
            && remote_name_in_play(rt, symbolic_remotes)
            && !is_meta_stack_branch)
            .then(|| rt.clone())
    }));
    if cg.explicit_seeds {
        claimed_remote_names.extend(cg.seeds.iter().filter_map(|t| {
            t.ref_name
                .clone()
                .filter(|r| r.as_ref().category() == Some(Category::RemoteBranch))
        }));
    }
    claimed_remote_names
}

/// The per-tip minting decision: the tip's commit run (a float's displaced build-time name
/// riding on the tip commit as a passive ref) and its planned name — `None` for floated/anonymized
/// tips, which start ANONYMOUS.
pub(super) fn tip_run_and_name(
    cg: &CommitGraph,
    tip: gix::ObjectId,
    in_set: &IdSet,
    boundaries: &IdSet,
    plan: &ChainPlan,
    visit: impl FnMut(usize),
) -> (Vec<Commit>, Option<(gix::refs::FullName, gix::ObjectId)>) {
    let is_boundary = |c: gix::ObjectId| boundaries.contains(&c);
    let float = plan.floats.iter().find(|fl| fl.tip == tip);
    let mut commits = commit_run(cg, tip, in_set, &is_boundary, visit);
    let named = plan.tip_name(tip);
    if let Some(displaced) = float.and_then(|fl| fl.displaced_ref_name.as_ref())
        && let Some(c0) = commits.first_mut()
        && !c0.refs.iter().any(|r| r.ref_name == *displaced)
    {
        c0.refs.push(RefInfo {
            ref_name: displaced.clone(),
            commit_id: Some(tip),
            worktree: None,
        });
        c0.refs.sort_by(|a, b| a.ref_name.cmp(&b.ref_name));
    }
    (commits, named)
}

/// The first-parent commit run owned by `tip`: `tip` and each first-parent descendant-in-history until
/// the next boundary (exclusive) or the set edge. `visit` sees each member's handle.
pub(super) fn commit_run(
    cg: &CommitGraph,
    tip: gix::ObjectId,
    in_set: &IdSet,
    is_boundary: &impl Fn(gix::ObjectId) -> bool,
    mut visit: impl FnMut(usize),
) -> Vec<Commit> {
    // Handle space: one `by_id` lookup for the tip, plain vec reads after.
    let mut out = Vec::new();
    let mut cursor = cg.index_of(tip);
    while let Some(c) = cursor {
        let id = cg.id_at(c);
        if !in_set.contains(&id) {
            break;
        }
        if id != tip && is_boundary(id) {
            break;
        }
        visit(c);
        out.push(cg.node_at(c).commit.clone());
        cursor = cg.first_parent_at(c);
    }
    out
}
