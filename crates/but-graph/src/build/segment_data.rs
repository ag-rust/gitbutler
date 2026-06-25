//! The build authors [`Segment`](crate::Segment)s directly: they are minted from the plan
//! data alone (allocation order, connection order, adjusted endpoints); the stored ref
//! positions derive from them
//! ([`derive_ref_positions`](super::ref_positions::derive_ref_positions)), and the finished
//! graph installs them as its storage ([`render_segment_graph`]) — position `i` is id `i`.

use std::collections::{BTreeMap, HashMap, HashSet};

use gix::reference::Category;

use super::chains::AdvancedOutside;
use super::materialize::{commit_run, tip_run_and_name};
use super::plan::ChainPlan;
use super::remotes::{AheadRegion, region_tips, remote_name_in_play, unique_plain_local};
use super::{IdMap, IdSet, is_plain_local_branch};
use crate::ref_layout::{GroupPlacement, RefLayout};
use crate::segment_graph::Connection;
use crate::{Commit, CommitGraph};

/// The segment arena the build authors: indices are allocation-ordered and no pass removes
/// entries, so a segment's position IS its id. The build works on the very
/// [`Segment`](crate::Segment)s the finished graph holds — render merely installs them.
#[derive(Default)]
pub(super) struct SegmentData {
    pub segments: Vec<crate::Segment>,
    /// The entrypoint's segment — the root of the layout's reach computation, and its name marks
    /// the HEAD ordinals.
    pub entrypoint_sidx: Option<usize>,
    /// The commit the entrypoint rests on, `None` when unborn.
    pub entrypoint: Option<gix::ObjectId>,
}

impl SegmentData {
    /// The entrypoint in the graph's representation: the segment (== segment id) plus the tip
    /// commit, or [`Unborn`](crate::EntryPointCommit::Unborn) when the entrypoint ref has none.
    pub(super) fn graph_entrypoint(&self) -> Option<(usize, crate::EntryPointCommit)> {
        self.entrypoint_sidx.map(|sidx| {
            (
                sidx,
                self.entrypoint
                    .map(crate::EntryPointCommit::AtCommit)
                    .unwrap_or(crate::EntryPointCommit::Unborn),
            )
        })
    }

    fn add_segment(&mut self, name: Option<gix::refs::FullName>, commits: Vec<Commit>) -> usize {
        let id = self.segments.len();
        self.segments.push(crate::Segment {
            id,
            // The `commit_id` is `None` for synthetic (metadata-derived) empties, whose name
            // has no resolved ref tip, and the worktree stays unfilled until [`enrich`].
            ref_info: name.map(|ref_name| crate::RefInfo {
                ref_name,
                commit_id: None,
                worktree: None,
            }),
            remote_tracking_ref_name: None,
            sibling_segment_id: None,
            remote_tracking_branch_segment_id: None,
            commits,
            // Filled by [`enrich`] once the names are final.
            metadata: None,
            connections: Vec::new(),
        });
        id
    }

    /// Record the commit `segment`'s name resolves to. A no-op on nameless segments.
    fn set_tip(&mut self, sidx: usize, id: gix::ObjectId) {
        if let Some(ri) = &mut self.segments[sidx].ref_info {
            ri.commit_id = Some(id);
        }
    }

    /// Name (or rename) `segment`, with the tip its name resolves to.
    fn set_name(
        &mut self,
        sidx: usize,
        ref_name: gix::refs::FullName,
        commit_id: Option<gix::ObjectId>,
    ) {
        self.segments[sidx].ref_info = Some(crate::RefInfo {
            ref_name,
            commit_id,
            worktree: None,
        });
    }

    /// Link a just-created remote-named segment to the local segment named by its tracking counterpart:
    /// the remote's sibling points at the local, and the local carries the remote's name and
    /// segment id. A no-op when no such local exists.
    fn link_remote_to_local(
        &mut self,
        remote_sidx: usize,
        remote_ref: &gix::refs::FullName,
        remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
    ) {
        let Some(local_name) = remote_tracking
            .iter()
            .find_map(|(l, r)| (r == remote_ref).then_some(l))
        else {
            return;
        };
        let Some(local_sidx) = self.sidx_by_ref(local_name) else {
            return;
        };
        self.segments[remote_sidx].sibling_segment_id = Some(local_sidx);
        self.segments[local_sidx].remote_tracking_ref_name = Some(remote_ref.clone());
        self.segments[local_sidx].remote_tracking_branch_segment_id = Some(remote_sidx);
    }

    /// Append `src` → `dst` with adjusted endpoints.
    fn connect(&mut self, src: usize, dst: usize) {
        let edge = self.adjusted_edge(src, dst);
        self.segments[src].connections.push(edge);
    }

    fn adjusted_edge(&self, src: usize, dst: usize) -> Connection {
        Connection::new(
            dst,
            self.segments[src].commits.last().map(|c| c.id),
            self.segments[dst].commits.first().map(|c| c.id),
        )
    }

    /// Insert `src` → `dst` at `slot` among `src`'s edges (clamped).
    fn insert_connect_at(&mut self, src: usize, slot: usize, dst: usize) {
        let edge = self.adjusted_edge(src, dst);
        let edges = &mut self.segments[src].connections;
        let slot = slot.min(edges.len());
        edges.insert(slot, edge);
    }

    /// Re-point `src`'s edges at `old_target` to `new_target`, clearing their destination
    /// endpoint (surgery invalidates it).
    fn retarget_edges(&mut self, src: usize, old_target: usize, new_target: usize) -> usize {
        let mut retargeted = 0;
        for edge in &mut self.segments[src].connections {
            if edge.target == old_target {
                edge.target = new_target;
                edge.dst_id = None;
                retargeted += 1;
            }
        }
        retargeted
    }

    fn sidx_by_commit(&self, commit: gix::ObjectId) -> Option<usize> {
        (0..self.segments.len())
            .find(|&sidx| self.segments[sidx].commits.iter().any(|c| c.id == commit))
    }

    /// Like [`Self::sidx_by_commit`] but ignoring `exclude`d segments — the pre-chain coverage view.
    fn sidx_by_commit_excluding(
        &self,
        commit: gix::ObjectId,
        exclude: &HashSet<usize>,
    ) -> Option<usize> {
        (0..self.segments.len()).find(|&sidx| {
            !exclude.contains(&sidx) && self.segments[sidx].commits.iter().any(|c| c.id == commit)
        })
    }

    /// The first segment (in id order) named `name`.
    fn sidx_by_ref(&self, name: &gix::refs::FullName) -> Option<usize> {
        (0..self.segments.len()).find(|&sidx| self.segments[sidx].ref_name() == Some(name.as_ref()))
    }

    /// The first segment (in id order) whose FIRST commit is `commit` — the region-reuse and
    /// existing-remote-tip probe.
    fn sidx_with_first_commit(&self, commit: gix::ObjectId) -> Option<usize> {
        (0..self.segments.len()).find(|&sidx| {
            self.segments[sidx]
                .commits
                .first()
                .is_some_and(|c| c.id == commit)
        })
    }

    fn is_remote_segment(&self, sidx: usize) -> bool {
        self.segments[sidx]
            .ref_name()
            .is_some_and(|name| name.category() == Some(Category::RemoteBranch))
    }

    /// Every segment's outgoing edges as target ordinals in FINAL parent order: real parents by
    /// their index in the source commit's parent array, commit-less edges after them in edge
    /// order, ordinals compacted by push order.
    pub(super) fn parent_ordered_targets(&self) -> OrderedTargets {
        let mut start = Vec::with_capacity(self.segments.len() + 1);
        let mut flat = Vec::new();
        let mut ordered: Vec<(usize, usize)> = Vec::new();
        start.push(0);
        for sidx in &self.segments {
            let mut empty_branch_count = 0usize;
            ordered.clear();
            for edge in &sidx.connections {
                // An edge's `src_id` is one of ITS OWN segment's commits (the last, by
                // construction) — found by scanning backwards, not via a table over
                // every commit in the graph.
                let edge_parents = edge.src_id.and_then(|src| {
                    sidx.commits
                        .iter()
                        .rev()
                        .find(|c| c.id == src)
                        .map(|c| c.parent_ids.as_slice())
                });
                debug_assert_eq!(
                    edge_parents.is_some(),
                    edge.src_id.is_some(),
                    "an edge's src_id is one of its own segment's commits"
                );
                let real_parent_index = edge_parents
                    .zip(edge.dst_id)
                    .and_then(|(parents, dst)| parents.iter().position(|p| *p == dst));
                let ordinal = match real_parent_index {
                    Some(idx) => idx,
                    None => {
                        let o = edge_parents.map_or(0, |p| p.len()) + empty_branch_count;
                        empty_branch_count += 1;
                        o
                    }
                };
                ordered.push((ordinal, edge.target));
            }
            ordered.sort_by_key(|(ordinal, _)| *ordinal);
            flat.extend(ordered.iter().map(|&(_, t)| t));
            start.push(flat.len());
        }
        OrderedTargets { start, flat }
    }
}

/// [`SegmentData::parent_ordered_targets`] in CSR form: one flat target list, segment-sliced.
pub(super) struct OrderedTargets {
    /// Segment `s`'s targets are `flat[start[s]..start[s + 1]]`.
    start: Vec<usize>,
    flat: Vec<usize>,
}

impl OrderedTargets {
    pub(super) fn of(&self, sidx: usize) -> &[usize] {
        &self.flat[self.start[sidx]..self.start[sidx + 1]]
    }
}

/// Everything the build reads: the commit graph plus the decided data (facts fields, the
/// plan, the layout, and the chain-structure decisions).
pub(super) struct GraphInputs<'a> {
    pub cg: &'a CommitGraph,
    pub tips: &'a [gix::ObjectId],
    pub in_set: &'a IdSet,
    pub boundaries: &'a IdSet,
    pub owner_of: &'a IdMap<gix::ObjectId>,
    pub plan: &'a ChainPlan,
    pub layout: &'a RefLayout,
    pub workspace_commit: gix::ObjectId,
    pub managed: bool,
    pub ws_empty_ref: Option<&'a gix::refs::FullName>,
    pub advanced_outside: &'a [AdvancedOutside],
    pub remote_tracking: &'a HashMap<gix::refs::FullName, gix::refs::FullName>,
    pub symbolic_remotes: &'a [String],
    pub stack_branches: Option<&'a [Vec<gix::refs::FullName>]>,
    pub region_pinned: &'a IdSet,
    pub claimed_remote_names: &'a HashSet<gix::refs::FullName>,
    pub entrypoint: gix::ObjectId,
    pub entrypoint_ref: Option<&'a gix::refs::FullName>,
    pub target_ref: Option<&'a gix::refs::FullName>,
    pub extra_target: Option<gix::ObjectId>,
}

/// THE BUILD: every materializer pass (mint + connect + chain structure + the remote passes +
/// the coverage regions + the sweeps) run on the store from the decisions alone. The store
/// authors the stored positions and renders verbatim into the segment graph.
#[tracing::instrument(name = "segment_data::build", level = "trace", skip_all)]
pub(super) fn build(inputs: GraphInputs<'_>) -> SegmentData {
    let (mut store, sidx_of_tip) = mint_segments(
        inputs.cg,
        inputs.tips,
        inputs.in_set,
        inputs.boundaries,
        inputs.plan,
        inputs.workspace_commit,
        inputs.remote_tracking,
    );
    let before_chains = store.segments.len();
    if inputs.managed {
        build_chain_structure(
            &mut store,
            &sidx_of_tip,
            inputs.workspace_commit,
            inputs.ws_empty_ref,
            inputs.advanced_outside,
            inputs.layout,
            inputs.remote_tracking,
        );
    }
    // The coverage gates evaluate the PRE-CHAIN view: segments minted by the chain-structure
    // pass don't count as coverage.
    let chain_created: HashSet<usize> = (before_chains..store.segments.len()).collect();
    let mut pending_edges: Vec<(usize, gix::ObjectId)> = Vec::new();
    add_remote_segments(
        inputs.cg,
        &mut store,
        &sidx_of_tip,
        inputs.in_set,
        inputs.owner_of,
        inputs.symbolic_remotes,
        inputs.stack_branches,
        inputs.region_pinned,
        inputs.remote_tracking,
        inputs.plan,
        inputs.claimed_remote_names,
        &mut pending_edges,
    );
    add_untracked_remote_segments(
        inputs.cg,
        &mut store,
        inputs.remote_tracking,
        &sidx_of_tip,
        inputs.in_set,
        inputs.owner_of,
    );
    surface_target_remote(
        inputs.cg,
        &mut store,
        inputs.target_ref,
        inputs.in_set,
        &sidx_of_tip,
        inputs.owner_of,
        inputs.plan,
        inputs.remote_tracking,
        inputs.region_pinned,
        inputs.claimed_remote_names,
        &mut pending_edges,
    );
    // The extra-target twin: a stored target position uncovered by any pre-chain segment grows
    // its own (nameless) region.
    if let Some(extra) = inputs.extra_target
        && inputs.cg.node(extra).is_some()
        && store
            .sidx_by_commit_excluding(extra, &chain_created)
            .is_none()
    {
        segment_ahead_region(
            inputs.cg,
            &mut store,
            None,
            extra,
            inputs.in_set,
            &sidx_of_tip,
            inputs.owner_of,
            inputs.remote_tracking,
            None,
            inputs.region_pinned,
            inputs.claimed_remote_names,
            &mut pending_edges,
        );
    }
    // The outside-entrypoint twin: an adhoc checkout outside the workspace grows its region.
    if !inputs.in_set.contains(&inputs.entrypoint)
        && inputs.cg.node(inputs.entrypoint).is_some()
        && store
            .sidx_by_commit_excluding(inputs.entrypoint, &chain_created)
            .is_none()
    {
        segment_ahead_region(
            inputs.cg,
            &mut store,
            inputs.entrypoint_ref,
            inputs.entrypoint,
            inputs.in_set,
            &sidx_of_tip,
            inputs.owner_of,
            inputs.remote_tracking,
            None,
            inputs.region_pinned,
            inputs.claimed_remote_names,
            &mut pending_edges,
        );
    }
    cover_explicit_seeds(
        inputs.cg,
        &mut store,
        &chain_created,
        inputs.in_set,
        &sidx_of_tip,
        inputs.owner_of,
        inputs.remote_tracking,
        inputs.region_pinned,
        inputs.claimed_remote_names,
        &mut pending_edges,
    );
    // The target's remote segment may exist before its LOCAL got a segment (the local can materialize
    // from the extra-target region above) — link them like every other creator does.
    if let Some(tr) = inputs.target_ref
        && let Some(tr_sidx) = store.sidx_by_ref(tr)
    {
        store.link_remote_to_local(tr_sidx, tr, inputs.remote_tracking);
    }
    add_co_located_remote_empties(&mut store, inputs.remote_tracking);
    wire_pending_edges(&mut store, pending_edges);
    float_remote_named_checkout(
        &mut store,
        inputs.entrypoint,
        inputs.entrypoint_ref,
        inputs.workspace_commit,
    );
    if inputs.managed {
        drop_suppressed_tip_links(&mut store, inputs.plan, &sidx_of_tip);
    }
    store.entrypoint = Some(inputs.entrypoint);
    store.entrypoint_sidx = resolve_entrypoint_row(
        &mut store,
        inputs.ws_empty_ref,
        inputs.entrypoint,
        inputs.entrypoint_ref,
        inputs.workspace_commit,
        inputs.remote_tracking,
    );
    strip_segment_named_refs(&mut store);
    store
}

/// Tip segments in facts order, float placeholders, then the parent edges — every decision read
/// from the plan data.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(level = "trace", skip_all)]
fn mint_segments(
    cg: &CommitGraph,
    tips: &[gix::ObjectId],
    in_set: &IdSet,
    boundaries: &IdSet,
    plan: &ChainPlan,
    workspace_commit: gix::ObjectId,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
) -> (SegmentData, IdMap<usize>) {
    let mut store = SegmentData::default();
    let mut sidx_of_tip: IdMap<usize> = IdMap::default();
    // Handle space: which segment owns each arena commit, and each run's bottom — filled by the
    // run walks, so the edge pass below needs no per-parent hash lookups.
    let mut sidx_at: Vec<u32> = vec![u32::MAX; cg.node_count()];
    let mut bottom_at: Vec<usize> = Vec::with_capacity(tips.len());
    for (sidx, &tip) in tips.iter().enumerate() {
        let mut bottom = usize::MAX;
        let (commits, named) = tip_run_and_name(cg, tip, in_set, boundaries, plan, |c| {
            sidx_at[c] = sidx as u32;
            bottom = c;
        });
        bottom_at.push(bottom);
        let sidx = store.add_segment(named.as_ref().map(|(name, _)| name.clone()), commits);
        if let Some((name, commit_id)) = named {
            store.set_tip(sidx, commit_id);
            store.segments[sidx].remote_tracking_ref_name = remote_tracking.get(&name).cloned();
        }
        sidx_of_tip.insert(tip, sidx);
    }
    let mut placeholder_of: IdMap<usize> = IdMap::default();
    for float in &plan.floats {
        // Placeholders are synthetic: their name has no resolved ref tip here.
        let sidx = store.add_segment(Some(float.name.clone()), Vec::new());
        store.segments[sidx].remote_tracking_ref_name = remote_tracking.get(&float.name).cloned();
        placeholder_of.insert(float.tip, sidx);
    }
    let float_tips: IdSet = plan.floats.iter().map(|fl| fl.tip).collect();
    for (src, &tip) in tips.iter().enumerate() {
        // A tip is an in-set boundary, so its run has at least itself.
        debug_assert_ne!(bottom_at[src], usize::MAX);
        // One edge per in-graph parent of the run's bottom, in first-parent order; the
        // WORKSPACE commit's edge to a floated parent goes to the float's placeholder.
        for p in cg.connected_parents_at(bottom_at[src]) {
            let r = sidx_at[p];
            if r == u32::MAX {
                continue;
            }
            let dst = if tip == workspace_commit && float_tips.contains(&cg.id_at(p)) {
                placeholder_of[&cg.id_at(p)]
            } else {
                r as usize
            };
            store.connect(src, dst);
        }
    }
    for float in &plan.floats {
        let (Some(&ph), Some(&tip_sidx)) =
            (placeholder_of.get(&float.tip), sidx_of_tip.get(&float.tip))
        else {
            continue;
        };
        store.connect(ph, tip_sidx);
    }
    (store, sidx_of_tip)
}

/// The empty-workspace splice, the decided advanced-outside branches, then the store's
/// chains — all consumed as data.
#[tracing::instrument(level = "trace", skip_all)]
fn build_chain_structure(
    store: &mut SegmentData,
    sidx_of_tip: &IdMap<usize>,
    workspace_commit: gix::ObjectId,
    ws_empty_ref: Option<&gix::refs::FullName>,
    advanced_outside: &[AdvancedOutside],
    layout: &RefLayout,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
) {
    let mut ws_empty_sidx = None;
    if let Some(ws_ref) = ws_empty_ref
        && let Some(&stack) = sidx_of_tip.get(&workspace_commit)
    {
        let sidx = store.add_segment(Some(ws_ref.clone()), Vec::new());
        store.set_tip(sidx, workspace_commit);
        store.connect(sidx, stack);
        ws_empty_sidx = Some(sidx);
    }
    for decision in advanced_outside {
        let Some(owner) = store.sidx_by_commit(decision.rejoin) else {
            continue;
        };
        let sidx = store.add_segment(decision.name.clone(), decision.commits.clone());
        if let Some(name) = decision.name.as_ref() {
            store.set_tip(sidx, decision.tip);
            store.segments[sidx].remote_tracking_ref_name = remote_tracking.get(name).cloned();
            // Only a NAMED advanced branch is the in-workspace segment's sibling; the workspace
            // position itself never links to outside content.
            if decision.rejoin != workspace_commit
                && store.segments[owner].sibling_segment_id.is_none()
            {
                store.segments[owner].sibling_segment_id = Some(sidx);
            }
        }
        store.connect(sidx, owner);
    }
    let ws_sidx = ws_empty_sidx.or_else(|| sidx_of_tip.get(&workspace_commit).copied());
    insert_empty_branches(store, ws_sidx, layout, remote_tracking);
}

/// Anonymous bases lose their names, naming refs take their anchors, empties splice above in metadata
/// order.
#[tracing::instrument(level = "trace", skip_all)]
fn insert_empty_branches(
    store: &mut SegmentData,
    ws_sidx: Option<usize>,
    layout: &RefLayout,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
) {
    for &tip in &layout.anonymous_bases {
        let Some(anchor) = store.sidx_by_commit(tip) else {
            continue;
        };
        store.segments[anchor].ref_info = None;
        store.segments[anchor].remote_tracking_ref_name = None;
        store.segments[anchor].remote_tracking_branch_segment_id = None;
    }
    for (li, chain) in layout.chains.iter().enumerate() {
        let mut from_sidx = ws_sidx;
        for &(commit, gi) in &chain.anchors {
            let group = &layout.at_commit[&commit][gi];
            if group.placement == GroupPlacement::Skipped {
                continue;
            }
            let Some(anchor) = store.sidx_by_commit(commit) else {
                continue;
            };
            if let Some(naming_ref) = &group.naming_ref {
                store.set_name(anchor, naming_ref.name.clone(), Some(commit));
                store.segments[anchor].remote_tracking_ref_name =
                    remote_tracking.get(&naming_ref.name).cloned();
                if naming_ref.clear_remote {
                    store.segments[anchor].remote_tracking_branch_segment_id = None;
                }
            }
            if let GroupPlacement::Splice {
                members,
                into_owning_chain,
            } = &group.placement
                && !members.is_empty()
            {
                insert_empty_chain_above(
                    store,
                    from_sidx,
                    anchor,
                    members,
                    remote_tracking,
                    *into_owning_chain,
                    (from_sidx == ws_sidx).then_some(li),
                );
            }
            from_sidx = Some(anchor);
        }
    }
}

/// Splice a run of empty named segments in above `anchor`: incoming edges redirect to the top
/// empty (or a fresh edge from `from_sidx` joins it), and the bottom empty connects down.
#[allow(clippy::too_many_arguments)]
fn insert_empty_chain_above(
    store: &mut SegmentData,
    from_sidx: Option<usize>,
    anchor: usize,
    empties: &[gix::refs::FullName],
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
    into_owning_chain: bool,
    fresh_connection_slot: Option<usize>,
) {
    let ids: Vec<usize> = empties
        .iter()
        .map(|b| {
            let sidx = store.add_segment(Some(b.clone()), Vec::new());
            store.segments[sidx].remote_tracking_ref_name = remote_tracking.get(b).cloned();
            sidx
        })
        .collect();
    let Some(&top) = ids.first() else {
        return;
    };
    if let Some(from_sidx) = from_sidx {
        let mut redirected = false;
        let redirect_sources: Vec<usize> = if into_owning_chain {
            (0..store.segments.len())
                .filter(|&sidx| !ids.contains(&sidx) && !store.is_remote_segment(sidx))
                .collect()
        } else {
            vec![from_sidx]
        };
        for source in redirect_sources {
            redirected |= store.retarget_edges(source, anchor, top) > 0;
        }
        if !redirected {
            let find_parent = |require_commits: bool| {
                (0..store.segments.len()).find(|&sidx| {
                    sidx != from_sidx
                        && !store.is_remote_segment(sidx)
                        && (!require_commits || !store.segments[sidx].commits.is_empty())
                        && store.segments[sidx]
                            .connections
                            .iter()
                            .any(|e| e.target == anchor)
                })
            };
            let chain_parent = into_owning_chain
                .then(|| find_parent(true).or_else(|| find_parent(false)))
                .flatten();
            match chain_parent {
                Some(parent) => {
                    store.retarget_edges(parent, anchor, top);
                }
                None => match fresh_connection_slot {
                    Some(slot) => store.insert_connect_at(from_sidx, slot, top),
                    None => store.connect(from_sidx, top),
                },
            }
        }
    }
    for i in 0..ids.len() {
        let next = ids.get(i + 1).copied().unwrap_or(anchor);
        store.connect(ids[i], next);
    }
}

/// The remote pass: locals keyed on the plan's pre-chain names in link order, behind/at
/// remotes as empty roots (skipped when the plan's rename already named the owner — that
/// owner still gets the sibling/tracking links), ahead remotes segmented region by region.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(level = "trace", skip_all)]
fn add_remote_segments(
    cg: &CommitGraph,
    store: &mut SegmentData,
    sidx_of_tip: &IdMap<usize>,
    in_set: &IdSet,
    owner_of: &IdMap<gix::ObjectId>,
    symbolic_remotes: &[String],
    stack_branches: Option<&[Vec<gix::refs::FullName>]>,
    region_pinned: &IdSet,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
    plan: &ChainPlan,
    claimed_remote_names: &HashSet<gix::refs::FullName>,
    pending_edges: &mut Vec<(usize, gix::ObjectId)>,
) {
    let mut locals: Vec<(usize, gix::refs::FullName)> = sidx_of_tip
        .iter()
        .filter_map(|(&tip, &sidx)| {
            let name = plan.base_name_of.get(&tip)?;
            let rt = remote_tracking.get(name).cloned()?;
            Some((store.sidx_by_ref(name).unwrap_or(sidx), rt))
        })
        .collect();
    locals.sort_by_key(|&(sidx, ..)| sidx);
    for (link_sidx, remote_ref) in locals {
        let Some(remote_tip) = cg.commit_by_ref(remote_ref.as_ref()) else {
            continue;
        };
        if in_set.contains(&remote_tip) {
            let owner = owner_of.get(&remote_tip).copied().unwrap_or(remote_tip);
            let owner_sidx = sidx_of_tip[&owner];
            let named_by_this = plan
                .renames
                .get(&owner)
                .is_some_and(|(name, _)| name == &remote_ref);
            if named_by_this {
                store.segments[owner_sidx].sibling_segment_id = Some(link_sidx);
                store.segments[link_sidx].remote_tracking_branch_segment_id = Some(owner_sidx);
            } else {
                let remote_sidx = store.add_segment(Some(remote_ref.clone()), Vec::new());
                store.set_tip(remote_sidx, remote_tip);
                store.segments[remote_sidx].sibling_segment_id = Some(link_sidx);
                store.segments[link_sidx].remote_tracking_branch_segment_id = Some(remote_sidx);
                store.connect(remote_sidx, owner_sidx);
            }
            continue;
        }
        let in_play = remote_name_in_play(&remote_ref, symbolic_remotes);
        let is_metadata_stack_branch = stack_branches
            .into_iter()
            .flatten()
            .flatten()
            .any(|b| *b == remote_ref);
        if !in_play || is_metadata_stack_branch {
            continue;
        }
        segment_ahead_region(
            cg,
            store,
            Some(&remote_ref),
            remote_tip,
            in_set,
            sidx_of_tip,
            owner_of,
            remote_tracking,
            Some(link_sidx),
            region_pinned,
            claimed_remote_names,
            pending_edges,
        );
    }
}

/// Segment one AHEAD region into segments: the [`AheadRegion`] shape, the interior cut/stop scan,
/// then segments and edges in creation order.
#[allow(clippy::too_many_arguments)]
fn segment_ahead_region(
    cg: &CommitGraph,
    store: &mut SegmentData,
    remote_ref: Option<&gix::refs::FullName>,
    remote_tip: gix::ObjectId,
    in_set: &IdSet,
    sidx_of_tip: &IdMap<usize>,
    owner_of: &IdMap<gix::ObjectId>,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
    local_sidx: Option<usize>,
    pinned_commits: &IdSet,
    claimed_remote_names: &HashSet<gix::refs::FullName>,
    pending_edges: &mut Vec<(usize, gix::ObjectId)>,
) {
    let region = AheadRegion::compute(cg, remote_tip, in_set);
    let ahead_set = &region.set;
    let is_boundary =
        |c: gix::ObjectId| region.is_shape_boundary(cg, remote_tip, pinned_commits, c);
    // Merge-heavy regions make nearly every commit a tip, and a per-tip store scan is
    // quadratic (20s on an 80k-commit repo) — index first commits up front instead.
    let mut first_commit_sidx: IdMap<usize> = IdMap::default();
    let mut remote_first_commits: IdSet = IdSet::default();
    for (sidx, seg) in store.segments.iter().enumerate() {
        if let Some(first) = seg.commits.first() {
            first_commit_sidx.entry(first.id).or_insert(sidx);
            if store.is_remote_segment(sidx) {
                remote_first_commits.insert(first.id);
            }
        }
    }

    let root_is_remote =
        remote_ref.is_some_and(|r| r.as_ref().category() == Some(Category::RemoteBranch));
    let mut interior_cuts: IdMap<gix::refs::FullName> = IdMap::default();
    let mut stop: Option<gix::ObjectId> = None;
    if root_is_remote {
        let existing_remote_tip = |c: gix::ObjectId| remote_first_commits.contains(&c);
        let mut id = cg
            .first_parent(remote_tip)
            .filter(|p| ahead_set.contains(p));
        while let Some(c) = id {
            if is_boundary(c) {
                break;
            }
            if cg
                .refs_at(c)
                .iter()
                .any(|r| claimed_remote_names.contains(r))
                || existing_remote_tip(c)
            {
                stop = Some(c);
                break;
            }
            if let Some(r) = cg.refs_at(c).into_iter().find(|r| {
                r.as_ref().category() == Some(Category::RemoteBranch)
                    && !claimed_remote_names.contains(r)
                    && store.sidx_by_ref(r).is_none()
            }) {
                interior_cuts.insert(c, r);
            }
            id = cg.first_parent(c).filter(|p| ahead_set.contains(p));
        }
    }
    let is_boundary =
        |c: gix::ObjectId| is_boundary(c) || interior_cuts.contains_key(&c) || stop == Some(c);

    let tips = region_tips(cg, &region, remote_tip, &is_boundary);
    let mut ahead_owner: IdMap<gix::ObjectId> = IdMap::default();
    let mut ahead_sidx: IdMap<usize> = IdMap::default();
    let mut reused: IdSet = IdSet::default();
    for &tip in &tips {
        if stop == Some(tip) {
            continue;
        }
        let commits = commit_run(cg, tip, ahead_set, &is_boundary, |_| {});
        for c in &commits {
            ahead_owner.insert(c.id, tip);
        }
        let is_root = tip == remote_tip;
        if !is_root && let Some(&existing) = first_commit_sidx.get(&tip) {
            ahead_sidx.insert(tip, existing);
            reused.insert(tip);
            continue;
        }
        let name = if is_root {
            remote_ref
                .cloned()
                .or_else(|| unique_plain_local(cg, remote_tip))
        } else {
            interior_cuts
                .get(&tip)
                .cloned()
                .or_else(|| unique_plain_local(cg, tip))
        };
        let sidx = store.add_segment(name, commits);
        if let Some(name) = store.segments[sidx].ref_name().map(|n| n.to_owned()) {
            store.set_tip(sidx, if is_root { remote_tip } else { tip });
            if is_plain_local_branch(&name) {
                store.segments[sidx].remote_tracking_ref_name = remote_tracking.get(&name).cloned();
            }
        }
        if is_root {
            store.segments[sidx].sibling_segment_id = local_sidx;
            if let Some(local_sidx) = local_sidx {
                store.segments[local_sidx].remote_tracking_branch_segment_id = Some(sidx);
            }
        }
        if let Some(cut_ref) = interior_cuts.get(&tip) {
            let cut_ref = cut_ref.clone();
            store.link_remote_to_local(sidx, &cut_ref, remote_tracking);
        }
        if let Some(first) = store.segments[sidx].commits.first() {
            first_commit_sidx.entry(first.id).or_insert(sidx);
        }
        ahead_sidx.insert(tip, sidx);
    }

    for &tip in &tips {
        if reused.contains(&tip) || stop == Some(tip) {
            continue;
        }
        let src = ahead_sidx[&tip];
        let bottom = store.segments[src]
            .commits
            .last()
            .map(|c| c.id)
            .unwrap_or(tip);
        for parent in cg.all_parent_ids(bottom) {
            let dst = if ahead_set.contains(&parent) {
                ahead_owner
                    .get(&parent)
                    .and_then(|o| ahead_sidx.get(o))
                    .copied()
            } else {
                owner_of
                    .get(&parent)
                    .and_then(|o| sidx_of_tip.get(o))
                    .copied()
            };
            if let Some(dst) = dst {
                store.connect(src, dst);
            } else if ahead_set.contains(&parent) {
                pending_edges.push((src, parent));
            }
        }
    }
}

/// Unclaimed remote refs whose local counterpart shares the tip become empty roots into the
/// owning segment (in-set case only).
#[tracing::instrument(level = "trace", skip_all)]
fn add_untracked_remote_segments(
    cg: &CommitGraph,
    store: &mut SegmentData,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
    sidx_of_tip: &IdMap<usize>,
    in_set: &IdSet,
    owner_of: &IdMap<gix::ObjectId>,
) {
    let mut remote_refs: std::collections::BTreeSet<gix::refs::FullName> =
        std::collections::BTreeSet::new();
    for c in cg.commit_ids() {
        for r in cg.refs_at(c) {
            if r.as_ref().category() == Some(Category::RemoteBranch) {
                remote_refs.insert(r);
            }
        }
    }
    for r in remote_refs {
        if store.sidx_by_ref(&r).is_some() {
            continue;
        }
        let Some(tip) = cg.commit_by_ref(r.as_ref()) else {
            continue;
        };
        let has_local_counterpart = cg
            .refs_at(tip)
            .iter()
            .any(|l| remote_tracking.get(l) == Some(&r));
        if !has_local_counterpart {
            continue;
        }
        if in_set.contains(&tip)
            && let Some(&owner) = owner_of.get(&tip)
            && let Some(&owner_sidx) = sidx_of_tip.get(&owner)
        {
            let remote_sidx = store.add_segment(Some(r.clone()), Vec::new());
            store.set_tip(remote_sidx, tip);
            store.connect(remote_sidx, owner_sidx);
            store.link_remote_to_local(remote_sidx, &r, remote_tracking);
        }
    }
}

/// Surface the target remote: the in-set case adds the sibling link when the plan's rename
/// already named the owner; an outside target grows its region, and a local tracking branch
/// on the region's tip takes the name — the remote becomes an empty segment above it,
/// sibling-linked.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(level = "trace", skip_all)]
fn surface_target_remote(
    cg: &CommitGraph,
    store: &mut SegmentData,
    target_ref: Option<&gix::refs::FullName>,
    in_set: &IdSet,
    sidx_of_tip: &IdMap<usize>,
    owner_of: &IdMap<gix::ObjectId>,
    plan: &ChainPlan,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
    region_pinned: &IdSet,
    claimed_remote_names: &HashSet<gix::refs::FullName>,
    pending_edges: &mut Vec<(usize, gix::ObjectId)>,
) {
    let Some(tr) = target_ref else { return };
    if tr.as_ref().category() != Some(Category::RemoteBranch) {
        return;
    }
    let Some(tip) = cg.commit_by_ref(tr.as_ref()) else {
        return;
    };
    if in_set.contains(&tip) {
        let owner_tip = owner_of.get(&tip).copied().unwrap_or(tip);
        if plan.renames.get(&owner_tip).is_some_and(|(n, _)| n == tr)
            && let Some(owner_sidx) = store.sidx_by_commit(tip)
        {
            // Sibling: the segment whose FIRST commit is the local tracking ref's position.
            let local_sidx = remote_tracking
                .iter()
                .find(|(_, r)| *r == tr)
                .and_then(|(local, _)| cg.commit_by_ref(local.as_ref()))
                .and_then(|lc| {
                    store.sidx_by_commit(lc).filter(|&sidx| {
                        sidx != owner_sidx
                            && store.segments[sidx]
                                .commits
                                .first()
                                .is_some_and(|c| c.id == lc)
                    })
                });
            if let Some(local_sidx) = local_sidx {
                store.segments[owner_sidx].sibling_segment_id = Some(local_sidx);
            }
        }
        return;
    }
    if store.sidx_by_ref(tr).is_some() {
        return;
    }
    segment_ahead_region(
        cg,
        store,
        Some(tr),
        tip,
        in_set,
        sidx_of_tip,
        owner_of,
        remote_tracking,
        None,
        region_pinned,
        claimed_remote_names,
        pending_edges,
    );
    let local_on_tip = remote_tracking
        .iter()
        .find(|(local, r)| *r == tr && cg.commit_by_ref(local.as_ref()) == Some(tip))
        .map(|(local, _)| local.clone());
    if let Some(local) = local_on_tip
        && let Some(owner) = store.sidx_by_commit(tip)
        && store.segments[owner].ref_name() == Some(tr.as_ref())
        && store.segments[owner]
            .commits
            .first()
            .is_some_and(|c| c.id == tip)
    {
        store.set_name(owner, local, Some(tip));
        store.segments[owner].remote_tracking_ref_name = Some(tr.clone());
        let remote_sidx = store.add_segment(Some(tr.clone()), Vec::new());
        store.set_tip(remote_sidx, tip);
        store.segments[remote_sidx].sibling_segment_id = Some(owner);
        store.segments[owner].remote_tracking_branch_segment_id = Some(remote_sidx);
        store.connect(remote_sidx, owner);
    }
}

/// An uncovered explicit seed grows its own region; a covered one whose ref names no segment gets
/// an empty tip-named segment above its owner.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(level = "trace", skip_all)]
fn cover_explicit_seeds(
    cg: &CommitGraph,
    store: &mut SegmentData,
    chain_created: &HashSet<usize>,
    in_set: &IdSet,
    sidx_of_tip: &IdMap<usize>,
    owner_of: &IdMap<gix::ObjectId>,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
    region_pinned: &IdSet,
    claimed_remote_names: &HashSet<gix::refs::FullName>,
    pending_edges: &mut Vec<(usize, gix::ObjectId)>,
) {
    for t in cg.seeds.iter().filter(|_| cg.explicit_seeds) {
        if cg.node(t.id).is_none() {
            continue;
        }
        match store.sidx_by_commit_excluding(t.id, chain_created) {
            None => segment_ahead_region(
                cg,
                store,
                t.ref_name.as_ref(),
                t.id,
                in_set,
                sidx_of_tip,
                owner_of,
                remote_tracking,
                None,
                region_pinned,
                claimed_remote_names,
                pending_edges,
            ),
            Some(owner_sidx) => {
                let Some(ref_name) = t.ref_name.clone() else {
                    continue;
                };
                if but_core::is_workspace_ref_name(ref_name.as_ref()) {
                    continue;
                }
                if store.sidx_by_ref(&ref_name).is_some()
                    || store.segments[owner_sidx].ref_name() == Some(ref_name.as_ref())
                {
                    continue;
                }
                // The plan deliberately left this tip-started segment anonymous — keep it that way.
                if store.segments[owner_sidx]
                    .commits
                    .first()
                    .is_some_and(|c| c.id == t.id)
                    && store.segments[owner_sidx].ref_info.is_none()
                {
                    continue;
                }
                let empty = store.add_segment(Some(ref_name.clone()), Vec::new());
                store.set_tip(empty, t.id);
                store.segments[empty].remote_tracking_ref_name =
                    remote_tracking.get(&ref_name).cloned();
                store.connect(empty, owner_sidx);
            }
        }
    }
}

/// Every further remote ref on a remote segment's first commit becomes an empty segment pointing at
/// it.
#[tracing::instrument(level = "trace", skip_all)]
fn add_co_located_remote_empties(
    store: &mut SegmentData,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
) {
    let existing = store.segments.len();
    for sidx in 0..existing {
        if !store.is_remote_segment(sidx) {
            continue;
        }
        let Some(first) = store.segments[sidx].commits.first().cloned() else {
            continue;
        };
        for ri in &first.refs {
            if ri.ref_name.as_ref().category() != Some(Category::RemoteBranch)
                || store.sidx_by_ref(&ri.ref_name).is_some()
            {
                continue;
            }
            let empty = store.add_segment(Some(ri.ref_name.clone()), Vec::new());
            store.set_tip(empty, first.id);
            store.connect(empty, sidx);
            store.link_remote_to_local(empty, &ri.ref_name, remote_tracking);
        }
    }
}

/// Connect each stopped segment to the segment owning its parent commit — every creator has run by
/// now, so the owner exists.
#[tracing::instrument(level = "trace", skip_all)]
fn wire_pending_edges(store: &mut SegmentData, pending_edges: Vec<(usize, gix::ObjectId)>) {
    for (src, parent) in pending_edges {
        let Some(dst) = store.sidx_by_commit(parent) else {
            continue;
        };
        store.connect(src, dst);
    }
}

/// A ref-less checkout at a remote-named segment's tip moves the name (and its links) to a fresh
/// empty segment above it; edges and links aimed at the named segment follow.
#[tracing::instrument(level = "trace", skip_all)]
fn float_remote_named_checkout(
    store: &mut SegmentData,
    entrypoint: gix::ObjectId,
    entrypoint_ref: Option<&gix::refs::FullName>,
    workspace_commit: gix::ObjectId,
) {
    if entrypoint_ref.is_some() || entrypoint == workspace_commit {
        return;
    }
    let Some(ep_sidx) = store.sidx_by_commit(entrypoint) else {
        return;
    };
    if !store.is_remote_segment(ep_sidx)
        || store.segments[ep_sidx].commits.first().map(|c| c.id) != Some(entrypoint)
    {
        return;
    }
    let ref_info = store.segments[ep_sidx].ref_info.take();
    let rt_name = store.segments[ep_sidx].remote_tracking_ref_name.take();
    let sibling = store.segments[ep_sidx].sibling_segment_id.take();
    let rt_row = store.segments[ep_sidx]
        .remote_tracking_branch_segment_id
        .take();
    let floated = store.add_segment(None, Vec::new());
    store.segments[floated].ref_info = ref_info;
    store.segments[floated].remote_tracking_ref_name = rt_name;
    store.segments[floated].sibling_segment_id = sibling;
    store.segments[floated].remote_tracking_branch_segment_id = rt_row;
    for sidx in 0..store.segments.len() {
        if sidx == floated {
            continue;
        }
        if store.segments[sidx].sibling_segment_id == Some(ep_sidx) {
            store.segments[sidx].sibling_segment_id = Some(floated);
        }
        if store.segments[sidx].remote_tracking_branch_segment_id == Some(ep_sidx) {
            store.segments[sidx].remote_tracking_branch_segment_id = Some(floated);
        }
        store.retarget_edges(sidx, ep_sidx, floated);
    }
    store.connect(floated, ep_sidx);
}

/// A floated or anonymized tip's build-time name lost its remote links to whichever segment finally
/// carries the name.
fn drop_suppressed_tip_links(
    store: &mut SegmentData,
    plan: &ChainPlan,
    sidx_of_tip: &IdMap<usize>,
) {
    for tip in plan
        .floats
        .iter()
        .map(|fl| fl.tip)
        .chain(plan.anonymous_bases.iter().copied())
    {
        if let Some(&sidx) = sidx_of_tip.get(&tip) {
            store.segments[sidx].remote_tracking_ref_name = None;
            store.segments[sidx].remote_tracking_branch_segment_id = None;
        }
    }
}

/// Pick the entrypoint's segment — the empty workspace segment for a ref-less checkout AT the
/// workspace position, the segment the checked-out ref already names, or the segment owning the
/// entrypoint commit (named/split on demand).
fn resolve_entrypoint_row(
    store: &mut SegmentData,
    ws_empty_ref: Option<&gix::refs::FullName>,
    entrypoint: gix::ObjectId,
    entrypoint_ref: Option<&gix::refs::FullName>,
    workspace_commit: gix::ObjectId,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
) -> Option<usize> {
    if let (Some(ws_sidx), None, true) = (
        ws_empty_ref.and_then(|r| store.sidx_by_ref(r)),
        entrypoint_ref,
        entrypoint == workspace_commit,
    ) {
        Some(ws_sidx)
    } else if let Some(named) = entrypoint_ref.and_then(|r| store.sidx_by_ref(r)) {
        Some(named)
    } else {
        name_entrypoint_segment(store, entrypoint, entrypoint_ref, remote_tracking)
    }
}

/// The segment whose FIRST commit is the entrypoint, named on demand — an anonymous segment takes the
/// checked-out ref's name; a segment named by ANOTHER ref gets an empty entrypoint-named segment
/// spliced in above (which becomes the pick).
fn name_entrypoint_segment(
    store: &mut SegmentData,
    entrypoint: gix::ObjectId,
    entrypoint_ref: Option<&gix::refs::FullName>,
    remote_tracking: &HashMap<gix::refs::FullName, gix::refs::FullName>,
) -> Option<usize> {
    let (sidx, pos) = (0..store.segments.len()).find_map(|sidx| {
        store.segments[sidx]
            .commits
            .iter()
            .position(|c| c.id == entrypoint)
            .map(|p| (sidx, p))
    })?;
    if pos != 0 {
        return None;
    }
    if let Some(ep_ref) = entrypoint_ref {
        match store.segments[sidx].ref_name().map(|n| n.to_owned()) {
            None => {
                store.set_name(sidx, ep_ref.clone(), Some(entrypoint));
                store.segments[sidx].remote_tracking_ref_name =
                    remote_tracking.get(ep_ref).cloned();
            }
            Some(existing) if existing != *ep_ref => {
                let empty = store.add_segment(Some(ep_ref.clone()), Vec::new());
                store.set_tip(empty, entrypoint);
                store.segments[empty].remote_tracking_ref_name =
                    remote_tracking.get(ep_ref).cloned();
                for other in 0..store.segments.len() {
                    if other == empty {
                        continue;
                    }
                    store.retarget_edges(other, sidx, empty);
                }
                store.connect(empty, sidx);
                return Some(empty);
            }
            Some(_) => {}
        }
    }
    Some(sidx)
}

/// A ref that names a segment (or is a named segment's remote-tracking counterpart) leaves every
/// commit's own ref list, as does every remote-category ref.
#[tracing::instrument(level = "trace", skip_all)]
fn strip_segment_named_refs(store: &mut SegmentData) {
    let names: HashSet<gix::refs::FullName> = store
        .segments
        .iter()
        .flat_map(|r| {
            r.ref_name()
                .map(|n| n.to_owned())
                .into_iter()
                .chain(r.remote_tracking_ref_name.clone())
        })
        .collect();
    for sidx in &mut store.segments {
        for commit in &mut sidx.commits {
            commit.refs.retain(|ri| {
                !names.contains(&ri.ref_name)
                    && ri.ref_name.as_ref().category() != Some(Category::RemoteBranch)
            });
        }
    }
}

/// The final enrichments, both pure functions of a name: worktree annotation (segment names and
/// the refs riding commits) and metadata classification. Runs once the names are final —
/// after the build AND the ad-hoc replay, which can still rename segments.
pub(super) fn enrich<T: but_core::RefMetadata>(
    store: &mut SegmentData,
    meta: &T,
    worktree_by_branch: &BTreeMap<gix::refs::FullName, Vec<crate::Worktree>>,
) {
    for sidx in &mut store.segments {
        if let Some(ri) = &mut sidx.ref_info {
            ri.worktree = worktree_by_branch
                .get(&ri.ref_name)
                .and_then(|w| w.first())
                .cloned();
        }
        sidx.metadata = sidx
            .ref_info
            .as_ref()
            .and_then(|ri| super::segment_metadata(ri.ref_name.as_ref(), meta));
        for commit in &mut sidx.commits {
            for ri in &mut commit.refs {
                if ri.worktree.is_none()
                    && let Some(wt) = worktree_by_branch.get(&ri.ref_name).and_then(|w| w.first())
                {
                    ri.worktree = Some(wt.clone());
                }
            }
        }
    }
}

/// Replay the ad-hoc same-tip rebuild on the store — the NON-managed path's post-build pass.
/// The decision (which persisted-order refs share the bottom tip) is repo+meta work recorded
/// as `orders`; the structural rewrite happens here, and the graph re-renders after.
pub(super) fn replay_ad_hoc(
    store: &mut SegmentData,
    cg: &CommitGraph,
    orders: &[Vec<gix::refs::FullName>],
    entrypoint_ref: Option<&gix::refs::FullName>,
) {
    for matching_refs in orders {
        ad_hoc_stack_rebuild(cg, store, matching_refs, entrypoint_ref);
    }
}

/// Install the store's segments as `graph`'s storage — the segments move over as they are,
/// and only the graph-level state is derived: the incoming index (one source entry per
/// connection, in id order) and the entrypoint.
#[tracing::instrument(level = "trace", skip_all)]
pub(super) fn render_segment_graph(graph: &mut crate::Graph, store: SegmentData) {
    graph.entrypoint = store.graph_entrypoint();
    graph.free.clear();
    graph.incoming = vec![Vec::new(); store.segments.len()];
    for (id, segment) in store.segments.iter().enumerate() {
        debug_assert_eq!(segment.id, id, "a segment's position is its id");
        for connection in &segment.connections {
            graph.incoming[connection.target].push(id);
        }
    }
    graph.segments = store.segments.into_iter().map(Some).collect();
}

/// Rebuild a same-tip chain by the persisted branch order: the bottom ordered ref owns the
/// shared tip, branches above become explicit empty segments chained on top, and outside edges
/// re-route through the (empty) top. The bottom tip resolves via `cg.commit_by_ref` — the
/// traversal read the same refs the repo holds.
fn ad_hoc_stack_rebuild(
    cg: &CommitGraph,
    store: &mut SegmentData,
    matching_refs: &[gix::refs::FullName],
    entrypoint_ref: Option<&gix::refs::FullName>,
) {
    let Some((bottom_ref, empty_refs)) = matching_refs.split_last() else {
        return;
    };
    if empty_refs.is_empty() {
        return;
    }
    let Some(bottom_sidx) = store.sidx_by_ref(bottom_ref).or_else(|| {
        cg.commit_by_ref(bottom_ref.as_ref())
            .and_then(|id| store.sidx_with_first_commit(id))
    }) else {
        return;
    };
    let bottom_tip = store.segments[bottom_sidx].commits.first().map(|c| c.id);
    store.set_name(bottom_sidx, bottom_ref.clone(), bottom_tip);
    if let Some(commit) = store.segments[bottom_sidx].commits.first_mut() {
        commit
            .refs
            .retain(|ri| !matching_refs.iter().any(|name| name == &ri.ref_name));
    }

    let mut empty_sidx_by_ref = BTreeMap::new();
    for (sidx, data) in store.segments.iter().enumerate() {
        if data.commits.is_empty()
            && let Some(name) = data.ref_name()
            && empty_refs.iter().any(|e| e.as_ref() == name)
        {
            empty_sidx_by_ref.insert(name.to_owned(), sidx);
        }
    }
    let mut ordered_sidxs: Vec<usize> = empty_refs
        .iter()
        .map(|name| {
            empty_sidx_by_ref
                .get(name)
                .copied()
                .unwrap_or_else(|| store.add_segment(Some(name.clone()), Vec::new()))
        })
        .collect();
    ordered_sidxs.push(bottom_sidx);
    let involved: HashSet<usize> = ordered_sidxs.iter().copied().collect();
    let top_sidx = ordered_sidxs[0];

    // Outside incoming edges re-point at the top in place.
    let retargets: Vec<(usize, usize)> = store
        .segments
        .iter()
        .enumerate()
        .filter(|(src, _)| !involved.contains(src))
        .flat_map(|(src, data)| {
            data.connections
                .iter()
                .filter(|edge| involved.contains(&edge.target))
                .map(move |edge| (src, edge.target))
        })
        .collect();
    for (src, old_target) in retargets {
        store.retarget_edges(src, old_target, top_sidx);
    }
    // Involved segments drop their own outgoing edges — except the bottom's leading outside — then
    // re-chain top-to-bottom. Each source ends with a single edge, so no first-parent re-sort.
    for &sidx in &involved {
        if sidx == bottom_sidx {
            store.segments[sidx]
                .connections
                .retain(|edge| !involved.contains(&edge.target));
        } else {
            store.segments[sidx].connections.clear();
        }
    }
    for pair in ordered_sidxs.windows(2) {
        let [above, below] = pair else {
            continue;
        };
        store.connect(*above, *below);
    }
    // The entrypoint follow-along: a rebuilt bundle containing the entrypoint's segment re-points
    // it at the segment the checked-out ref now names.
    if store
        .entrypoint_sidx
        .is_some_and(|ep| involved.contains(&ep))
        && let Some(ep_ref) = entrypoint_ref
        && let Some(new_ep) = ordered_sidxs
            .iter()
            .copied()
            .find(|&sidx| store.segments[sidx].ref_name() == Some(ep_ref.as_ref()))
    {
        store.entrypoint_sidx = Some(new_ep);
        store.entrypoint = store.segments[bottom_sidx].commits.first().map(|c| c.id);
    }
}
