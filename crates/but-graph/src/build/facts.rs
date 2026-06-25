//! Phase 1 of gather-then-build: pure boundary/shape reads over the commit-graph.
//!
//! The scans here touch every commit in the window, so they run in HANDLE space
//! (plain node indices with dense per-node vectors) and convert to ids only at the [`Facts`]
//! boundary — `by_id` hashing and per-call parent allocation dominate profiles otherwise.

use super::{IdMap, IdSet, disambiguated_ref_at};

/// Everything the build decides BEFORE any segment exists — pure facts over the commit graph,
/// ref positions, and metadata. Phase 1 of gather-then-build: the materialization and every
/// later pass read these; nothing here reads a segment.
pub(super) struct Facts {
    /// The commit set the LOCAL segments span.
    pub(super) in_set: IdSet,
    /// Is the checked-out workspace commit a real GitButler-managed merge?
    pub(super) ws_is_managed_merge: bool,
    /// Managed, but the ws ref sits on (or advanced past) a plain commit: an empty workspace
    /// segment is spliced in above.
    pub(super) empty_ws_case: bool,
    /// Stored/extra target positions (and explicit seeds): segments must start there.
    pub(super) pinned_commits: IdSet,
    /// Commits that START a segment.
    pub(super) boundaries: IdSet,
    /// The in-set entrypoint was NOT a boundary on its own and is forced to start a segment
    /// (a checkout inside a stack). Its tip keeps the split's naming precedence: the checked-out
    /// ref first, then plain disambiguation.
    pub(super) entrypoint_forced_boundary: bool,
    /// Which boundary's first-parent run each in-set commit belongs to.
    pub(super) owner_of: IdMap<gix::ObjectId>,
    /// The boundaries in materialization order: workspace first, then descending generation, id.
    pub(super) tips: Vec<gix::ObjectId>,
}

#[tracing::instrument(level = "trace", skip_all)]
pub(super) fn facts<T: but_core::RefMetadata>(b: &super::BuildInputs<'_>, meta: &T) -> Facts {
    let &super::BuildInputs {
        cg,
        workspace_commit,
        entrypoint,
        target,
        remote_tracking,
        stack_branches,
        managed,
        project_meta,
        options,
        ..
    } = b;
    let n = cg.node_count();
    // The commit set the LOCAL segments span: everything reachable from the workspace commit, plus the
    // target's own history WHEN the target has a local branch (it is `NotInRemote`) — e.g. an
    // integrated `main` that sits outside the workspace. A remote-only target (ahead of its local, not
    // `NotInRemote`) is NOT added: it becomes a remote segment instead.
    let mut in_set_at = vec![false; n];
    {
        let mut stack: Vec<usize> = cg.index_of(workspace_commit).into_iter().collect();
        if let Some(t) = target
            && cg
                .node(t)
                .is_some_and(|n| n.commit.flags.contains(crate::CommitFlags::NotInRemote))
        {
            stack.extend(cg.index_of(t));
        }
        while let Some(c) = stack.pop() {
            if !in_set_at[c] {
                in_set_at[c] = true;
                stack.extend(cg.connected_parents_at(c));
            }
        }
    }
    let in_set_id = |id: gix::ObjectId| cg.index_of(id).is_some_and(|c| in_set_at[c]);

    // In-set children per commit, to detect branch points (a commit reached by >1 child).
    let mut children_at: Vec<Vec<usize>> = vec![Vec::new(); n];
    for c in 0..n {
        if !in_set_at[c] {
            continue;
        }
        for p in cg.connected_parents_at(c) {
            if in_set_at[p] {
                children_at[p].push(c);
            }
        }
    }

    // Where each remote-tracking branch rejoins the local graph: the first in-set commit along the
    // remote tip's first-parent spine. These are segment boundaries (the remote connects INTO them).
    // Only remotes whose LOCAL counterpart is itself in the graph count — a remote for a branch that
    // lives outside the workspace (e.g. `origin/A-middle` on an outside `A-middle`) is never surfaced,
    // so its spine crossing an in-set commit must not carve a spurious boundary there.
    // The TARGET's rejoin always counts — it is surfaced regardless of where (or whether) its local
    // branch sits; e.g. `main` inside the target's own ahead region must not stop `origin/main`'s
    // line from carving its meeting point with the workspace.
    let target_tip = project_meta
        .target_ref
        .as_ref()
        .and_then(|tr| cg.commit_by_ref(tr.as_ref()));
    // An entrypoint OUTSIDE the workspace (an adhoc checkout in a repository that has one)
    // rejoins like a remote: the first in-set commit on its first-parent spine is a boundary
    // its outside region connects INTO.
    let entrypoint_outside = (!in_set_id(entrypoint)).then_some(entrypoint);

    // EXPLICIT tips (from_seeds) can point anywhere, and validation requires a tip
    // id to be its segment's first commit — so each one is a boundary. Workspace-discovered builds
    // must not carve these: there they are ordinary interior commits.
    let tip_ids: IdSet = if cg.explicit_seeds {
        cg.seeds.iter().map(|t| t.id).collect()
    } else {
        IdSet::default()
    };
    let remote_rejoins: IdSet = remote_tracking
        .iter()
        .filter(|(local, _)| cg.commit_by_ref(local.as_ref()).is_some_and(in_set_id))
        .filter_map(|(_, r)| cg.commit_by_ref(r.as_ref()))
        .chain(target_tip)
        .chain(entrypoint_outside)
        .filter_map(|tip| cg.first_on_spine(tip, |c| in_set_at[c]))
        .collect();

    // Is the checked-out workspace commit a real GitButler-managed merge, or a plain commit the ws ref
    // merely sits on (co-located with a stack tip) or has advanced PAST (an "on-top" commit above the
    // real merge)? Only a real merge is held in the workspace segment with its parents as stack tips;
    // otherwise the workspace segment is empty and spliced in above, and the commit keeps its normal
    // history and segmentation.
    let ws_is_managed_merge = managed && cg.is_managed_ws_commit(workspace_commit);
    let empty_ws_case = managed && !ws_is_managed_merge;

    // The workspace commit's parents are stack tips — always segment boundaries (so the workspace
    // segment holds only the workspace commit, even when a parent is anonymous, e.g. an advanced tip).
    // Only for a real managed merge; a plain checked-out tip, co-located stack tip, or advanced ref has
    // no stack parents to split on.
    let ws_parents: IdSet = if ws_is_managed_merge {
        cg.parents(workspace_commit).collect()
    } else {
        IdSet::default()
    };

    // A merge commit's segment holds only the merge, so its FIRST parent starts its own segment (the
    // second parent is already a boundary — reached by a non-first-parent edge).
    let mut merge_first_parent_at = vec![false; n];
    for c in 0..n {
        if in_set_at[c]
            && cg.connected_parent_count_at(c) > 1
            && let Some(p) = cg.first_parent_at(c)
            && in_set_at[p]
        {
            merge_first_parent_at[p] = true;
        }
    }

    // Every commit a workspace stack branch points at starts a segment: even when the commit is
    // name-ambiguous (several branches on it, so anonymous), the metadata branches float above it as
    // empty segments, so the commit itself must begin its own (anonymous) segment. A branch that
    // ADVANCED past the workspace anchors at its rejoin point instead — the first in-workspace commit
    // on its first-parent spine — which must equally start a segment (the advanced branch is projected
    // onto it via a sibling link).
    let metadata_commits: IdSet = stack_branches
        .unwrap_or(&[])
        .iter()
        .flatten()
        .filter_map(|b| cg.commit_by_ref(b.as_ref()))
        .filter_map(|tip| cg.first_on_spine(tip, |c| in_set_at[c]))
        .collect();

    // Stored/extra target positions must start their own segment: the projection's
    // `TargetCommit::from_commit` ignores a stored target commit that sits mid-segment, losing the
    // remembered base (and with it the workspace lower bound). Not restricted to the workspace set —
    // an older target position often sits inside the target REMOTE's ahead region.
    let mut pinned_commits: IdSet = project_meta
        .target_commit_id
        .into_iter()
        .chain(options.extra_target_commit_id)
        .filter(|&c| cg.node(c).is_some())
        .collect();
    // EXPLICIT tips split ahead regions too (e.g. an integrated target riding inside a remote's
    // ahead run must start its own segment there, like the walk's tip-seeded segments).
    pinned_commits.extend(tip_ids.iter().copied());

    // A commit starts a new segment when it carries a disambiguated ref, is the workspace tip, is a
    // merge, or is a convergence/branch point (reached by other than a single first-parent child).
    let is_boundary = |c: usize| -> bool {
        let id = cg.id_at(c);
        id == workspace_commit
            || ws_parents.contains(&id)
            || merge_first_parent_at[c]
            || remote_rejoins.contains(&id)
            || metadata_commits.contains(&id)
            || pinned_commits.contains(&id)
            || tip_ids.contains(&id)
            || cg.connected_parent_count_at(c) > 1
            || {
                let kids = children_at[c].as_slice();
                // Reached by a non-first-parent edge, or by more than one child.
                kids.len() > 1 || kids.iter().any(|&k| cg.first_parent_at(k) != Some(c))
            }
            // Last: the only check that touches refs and metadata.
            || disambiguated_ref_at(
                cg,
                c,
                remote_tracking,
                meta,
                Some(workspace_commit),
                project_meta.target_ref.as_ref(),
            )
            .is_some()
    };
    let mut boundary_at = vec![false; n];
    for c in 0..n {
        if in_set_at[c] && is_boundary(c) {
            boundary_at[c] = true;
        }
    }
    // A checkout inside a stack (from_tip): the entrypoint always starts its own
    // segment — planned here instead of splitting the enclosing segment after the build.
    let entrypoint_forced_boundary = cg
        .index_of(entrypoint)
        .filter(|&e| in_set_at[e])
        .is_some_and(|e| !std::mem::replace(&mut boundary_at[e], true));

    // Every boundary in the set starts a segment; each segment's commit run is the boundary plus its
    // first-parent tail up to (excluding) the next boundary. These runs partition the set, so assigning
    // each commit in a run to its boundary gives the owner directly — no reverse walk (a run's oldest
    // commit, e.g. a root, has no first-parent path back up to its own boundary).
    let mut owner_of: IdMap<gix::ObjectId> = IdMap::default();
    let mut tips_at: Vec<usize> = (0..n).filter(|&c| boundary_at[c]).collect();
    for &tip in &tips_at {
        let tip_id = cg.id_at(tip);
        let mut cursor = Some(tip);
        while let Some(c) = cursor {
            if c != tip && boundary_at[c] {
                break;
            }
            owner_of.insert(cg.id_at(c), tip_id);
            cursor = cg.first_parent_at(c).filter(|&p| in_set_at[p]);
        }
    }

    // Segment tips in a stable order (workspace first, then by descending generation, then id) so the
    // numbering is deterministic even though it need not match the walk's.
    tips_at.sort_by_key(|&t| {
        (
            cg.id_at(t) != workspace_commit,
            std::cmp::Reverse(cg.node_at(t).generation),
            cg.id_at(t),
        )
    });

    Facts {
        in_set: (0..n)
            .filter(|&c| in_set_at[c])
            .map(|c| cg.id_at(c))
            .collect(),
        ws_is_managed_merge,
        empty_ws_case,
        pinned_commits,
        boundaries: (0..n)
            .filter(|&c| boundary_at[c])
            .map(|c| cg.id_at(c))
            .collect(),
        entrypoint_forced_boundary,
        owner_of,
        tips: tips_at.iter().map(|&t| cg.id_at(t)).collect(),
    }
}
