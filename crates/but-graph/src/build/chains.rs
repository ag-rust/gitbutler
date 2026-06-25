//! Workspace chain-structure decisions: the empty-workspace splice name, the
//! advanced-outside branches, and the workspace lower bound.

use std::collections::{HashMap, HashSet};

use super::{IdMap, IdSet, disambiguated_ref, is_plain_local_branch};
use crate::{Commit, CommitGraph};

/// The name an empty-workspace splice carries. The traversal may have
/// dropped the special workspace ref from the commit's refs when a stack branch on the same
/// commit named its raw segment — the caller established the ref points here, so fall back to
/// the well-known name rather than silently skipping the workspace segment.
pub(super) fn empty_workspace_ref(
    cg: &CommitGraph,
    workspace_commit: gix::ObjectId,
) -> Option<gix::refs::FullName> {
    cg.refs_at(workspace_commit)
        .into_iter()
        .find(|r| but_core::is_workspace_ref_name(r.as_ref()))
        .or_else(|| but_core::WORKSPACE_REF_NAME.try_into().ok())
}

/// A metadata stack branch pointing at a commit OUTSIDE the workspace that has advanced past it,
/// decided as data: its outside run, the in-workspace commit it rejoins,
/// and its disambiguated name.
pub(super) struct AdvancedOutside {
    pub(super) tip: gix::ObjectId,
    pub(super) name: Option<gix::refs::FullName>,
    pub(super) commits: Vec<Commit>,
    pub(super) rejoin: gix::ObjectId,
}

/// The advanced-outside decisions, in metadata-stack order. `named` carries every ref name a
/// segment holds when the pass starts (mint names, float placeholders, the empty-workspace
/// splice) — a branch that already names a segment does not advance.
#[expect(clippy::too_many_arguments)]
pub(super) fn advanced_outside_decisions<T: but_core::RefMetadata>(
    cg: &CommitGraph,
    in_set: &IdSet,
    owner_of: &IdMap<gix::ObjectId>,
    stack_branches: Option<&[Vec<gix::refs::FullName>]>,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
    meta: &T,
    target_ref: Option<&gix::refs::FullName>,
    pinned_commits: &IdSet,
    mut named: HashSet<gix::refs::FullName>,
) -> Vec<AdvancedOutside> {
    let mut decisions: Vec<AdvancedOutside> = Vec::new();
    let mut outside_owned = IdSet::default();
    for b in stack_branches.into_iter().flatten().flatten() {
        // Only LOCAL branches advance past a workspace; metadata can also list remote refs as stack
        // branches, and those are handled by the remote passes.
        if !is_plain_local_branch(b) || named.contains(b) {
            continue;
        }
        let Some(tip) = cg.commit_by_ref(b.as_ref()) else {
            continue;
        };
        if in_set.contains(&tip) {
            continue;
        }
        // A PINNED commit (a stored/extra target position) must start its own segment via the
        // extra-target region — the projection derives the remembered base from it. When chains
        // run before the remote passes this pass would otherwise swallow it into the branch's
        // outside run first.
        if pinned_commits.contains(&tip) {
            continue;
        }
        // The branch's outside commits, down to where it rejoins the workspace.
        let mut commits: Vec<Commit> = Vec::new();
        let mut cursor = Some(tip);
        let mut rejoin = None;
        while let Some(id) = cursor {
            if in_set.contains(&id) {
                rejoin = Some(id);
                break;
            }
            if let Some(node) = cg.node(id) {
                commits.push(node.commit.clone());
            }
            cursor = cg.first_parent(id);
        }
        let (Some(rejoin), false) = (rejoin, commits.is_empty()) else {
            continue;
        };
        // Several stack branches can share the outside tip (e.g. an applied-branch preview where
        // `E` and `D` rest on the same not-yet-merged commit) — the run is materialized ONCE.
        if outside_owned.contains(&tip) || !owner_of.contains_key(&rejoin) {
            continue;
        }
        // Named like any tip: ambiguous refs keep the segment anonymous (the walk's floating
        // `►D, ►E` run), a unique branch names it (the advanced `B` above its own chain).
        let name = disambiguated_ref(cg, tip, remote_tracking, meta, None, target_ref);
        outside_owned.extend(commits.iter().map(|c| c.id));
        named.extend(name.clone());
        decisions.push(AdvancedOutside {
            tip,
            name,
            commits,
            rejoin,
        });
    }
    decisions
}

/// The lower bound the PROJECTION will use: the merge base with the target, extended DOWN to a
/// stored/extra target position lying below it — an older target location keeps the commits
/// integrated since then visible, so stacks resting between the bound and the merge base are real
/// (kept) stacks, not empty floats.
pub(super) fn effective_lower_bound(
    cg: &CommitGraph,
    workspace_commit: gix::ObjectId,
    target: Option<gix::ObjectId>,
    project_meta: &but_core::ref_metadata::ProjectMeta,
    options: &crate::walk::Options,
) -> Option<gix::ObjectId> {
    let mut lb = target
        .or(project_meta.target_commit_id)
        .or(options.extra_target_commit_id)
        .and_then(|t| cg.lowest_common_base(workspace_commit, t))?;
    for candidate in [
        project_meta.target_commit_id,
        options.extra_target_commit_id,
    ]
    .into_iter()
    .flatten()
    {
        if candidate != lb && cg.ancestor_set(lb).contains(&candidate) {
            lb = candidate;
        }
    }
    Some(lb)
}
