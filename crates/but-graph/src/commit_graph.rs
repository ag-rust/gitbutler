//! The commit graph: the raw traversal flattened into nodes — the substrate every
//! [`Graph`](crate::Graph) is built from (see `build`).
//!
//! # Why
//!
//! The pipeline is `gix traversal → CommitGraph → segment graph → projection`: the traversal
//! accumulates this table directly ([`CommitGraph::from_walk`]), the segment graph is rebuilt
//! from it, and but-rebase's editor adopts the carried copy as its mutable arena. The segment
//! layer turned out to be an *artifact of incremental construction*, not something either
//! consumer fundamentally needs:
//!
//! * **The rebase editor** is commit/ref-granular. It needs: commit id, parent ids (first-parent
//!   at `[0]`), the refs on each commit, an entrypoint, and parent-walk reachability. No segment
//!   boundaries, no segment ids. Its arena IS this type, with pick/ref side tables on top.
//!
//! * **Projection** emits segment-shaped output (`Stack`/`StackSegment`), but the segmentation is
//!   recomputable: a segment is a maximal first-parent run, split where a local-branch ref appears,
//!   at branch/merge points, and at the projection's own stops (entrypoint, merge-base, target).
//!   `generation`, merge-base, and remote-reachability are commit-level.
//!
//! So both build straight from the commit DAG, and segments are a *view* produced on the way to
//! projection rather than a stored graph.
//!
//! # The model
//!
//! A node is a commit (we reuse [`crate::Commit`]) plus its topological `generation`. An edge is
//! simply `commit → parent`, taken from `parent_ids` (first-parent at index 0) — there is no
//! `src`/`dst` within-segment payload to carry, because there are no segments to index into.
//!
//! A nice consequence: the **workspace commit's `parent_ids` array is the stack order**, so the
//! order-stacks machinery ([`crate::Graph`] post-pass) disappears — the order is read straight off
//! the merge commit's parents.
//!
//! Started as a standalone spike toward deleting the segment graph outright; the rebase side of
//! that unification has landed (the editor mutates this table and projects it), while read-side
//! consumers still see the segment graph rebuilt on top.

use gix::hashtable::{HashMap, HashSet};

use crate::{Commit, CommitFlags};

/// A node in the commit graph: a commit, plus where it sits topologically.
#[derive(Debug, Clone)]
pub struct CommitNode {
    /// The commit itself — `id`, `parent_ids` (first-parent at `[0]`), `flags`, and the `refs`
    /// pointing at it. This is exactly the per-commit data a segment carries.
    pub commit: Commit,
    /// Distance from a root (a commit with no parents in the graph). Higher means deeper in history.
    /// Used where the projection picks "the lowest of several tips"; cheap to compute during build.
    pub generation: u32,
}

/// One `commit → parent` edge, HANDLE-based: parallel to a slot in the commit's `parent_ids`.
/// The raw id in `parent_ids` is payload; this is the graph structure.
#[derive(Debug, Clone, Copy)]
struct ParentSlot {
    /// The parent's node, when it is present in the table. `None` = the raw parent id points
    /// outside the table (partial traversal — this subgraph roots here).
    target_node_idx: Option<usize>,
    /// Whether the traversal actually FOLLOWED this link. A parent can be present in the table
    /// via another path while this specific edge was severed (limits, integrated stop-early).
    connected: bool,
}

/// A commit-first table: nodes of commits with HANDLE-based `commit → parent` edges (one
/// `ParentSlot` per raw `parent_ids` entry) and the reverse (`parent → child`) adjacency derived
/// for downward walks. `ObjectId` is pure payload; `by_id` is a rebuildable lookup index.
#[derive(Debug, Clone, Default)]
pub struct CommitGraph {
    nodes: Vec<CommitNode>,
    by_id: HashMap<gix::ObjectId, usize>,
    /// Ref name → node holding it. Like `by_id` a rebuildable lookup, not truth: rebuilt
    /// whenever the nodes' attached refs are written (construction, [`Self::refresh_refs`])
    /// or node indices move ([`Self::compact`]).
    by_ref: std::collections::HashMap<gix::refs::FullName, usize>,
    /// Per node, one slot per `parent_ids` entry — presence and connectivity of that edge.
    parent_slots: Vec<Vec<ParentSlot>>,
    /// Rows the EDITOR removed (see the mutation surface). A tombstoned node keeps its arena
    /// index and (stale) payload; id-based reads skip it. Walk-built tables have none.
    tombstoned: Vec<bool>,
    /// `parent → children` adjacency, derived at build time so we can detect branch points and walk
    /// downward (the projection walks from the workspace tip toward the base).
    children: Vec<Vec<usize>>,
    /// Where traversal/HEAD started; the projection uses it as a focus boundary.
    entrypoint: Option<gix::ObjectId>,
    /// The ref the entrypoint was checked out as, if any. When set, it names the entrypoint segment
    /// (overriding disambiguation), mirroring `from_tip(id, Some(ref))`.
    entrypoint_ref: Option<gix::refs::FullName>,
    /// Commits whose message marks them as a GitButler-managed workspace commit. Kept out of
    /// [`CommitFlags`](crate::CommitFlags) so it neither perturbs the walk's goal bits nor the
    /// segment fingerprint; used to tell a real managed merge from a ws ref advanced past it.
    managed_ws_commits: HashSet<gix::ObjectId>,
    /// When built [from the walk](Self::from_walk): whether the traversal stopped queueing after
    /// hitting the hard limit. Derived tables must carry it onto the final `Graph`.
    pub(crate) hard_limit_hit: bool,
    /// When built [from the walk](Self::from_walk): the traversal's normalized seed tips. Graphs
    /// built from EXPLICIT tips must carry them onto the final `Graph` — the projection reads tip
    /// roles (e.g. integrated tips) for such graphs.
    pub(crate) seeds: Vec<crate::walk::Seed>,
    /// Built from EXPLICIT tips ([`Self::from_walk_seeds`]): every tip must start (or get) its own
    /// segment. Workspace-discovered builds must NOT carve boundaries at their normalized tips —
    /// there they are ordinary interior commits unless the plan makes them boundaries.
    pub(crate) explicit_seeds: bool,
    /// The ref placement table authored at build time (see [`ref_layout`](crate::ref_layout))
    /// and consumed by the builder's chain-structure pass. `None` until the table goes through
    /// the builder. Mirrors the refs the graph carried WHEN IT WAS BUILT — like
    /// [`Commit::refs`](crate::Commit), it goes stale under editor mutation and is re-authored
    /// by the write-through projection.
    pub(crate) layout: Option<crate::ref_layout::RefLayout>,
}

impl CommitGraph {
    /// The slots of `id`'s node, when present.
    fn slots_of(&self, id: gix::ObjectId) -> Option<&[ParentSlot]> {
        self.by_id
            .get(&id)
            .map(|&idx| self.parent_slots[idx].as_slice())
    }

    /// Assemble from the walk's outcome (see `walk::walker`): the walk's `by_id` is adopted
    /// as-is (collection order IS node order), and slot connectivity comes straight from the
    /// followed edges rather than being re-derived.
    #[tracing::instrument(name = "CommitGraph::from_walk_outcome", level = "trace", skip_all)]
    pub(crate) fn from_walk_outcome(o: crate::walk::walker::WalkOutcome) -> Self {
        let nodes: Vec<CommitNode> = o
            .commits
            .into_iter()
            .map(|commit| CommitNode {
                commit,
                generation: 0,
            })
            .collect();
        let by_id = o.by_id;
        let mut children = vec![Vec::new(); nodes.len()];
        let parent_slots: Vec<Vec<ParentSlot>> = nodes
            .iter()
            .enumerate()
            .map(|(idx, n)| {
                let followed = o.parents_followed.get(&n.commit.id);
                n.commit
                    .parent_ids
                    .iter()
                    .map(|parent| {
                        let target_node_idx = by_id.get(parent).copied();
                        let connected = followed.is_some_and(|ps| ps.contains(parent));
                        if connected && let Some(pidx) = target_node_idx {
                            children[pidx].push(idx);
                        }
                        ParentSlot {
                            target_node_idx,
                            connected,
                        }
                    })
                    .collect()
            })
            .collect();

        let tombstoned = vec![false; nodes.len()];
        let mut table = CommitGraph {
            nodes,
            by_id,
            by_ref: Default::default(),
            parent_slots,
            children,
            tombstoned,
            entrypoint: o.entrypoint,
            entrypoint_ref: o.entrypoint_ref,
            managed_ws_commits: HashSet::default(),
            hard_limit_hit: o.hard_limit_hit,
            seeds: o.seeds,
            explicit_seeds: false,
            layout: None,
        };
        table.recompute_generations();
        table.rebuild_by_ref();
        // Goal bits stop mattering the moment the traversal ends — and their numbering depends
        // on tip processing order, so tables from different builds could never compare equal
        // while carrying them.
        table.strip_goal_flags();
        table
    }

    /// Build by running the real traversal (queue, goals, limits, flag propagation), accumulating
    /// commits directly. This keeps the battle-tested traversal semantics — extents (limit cuts,
    /// integrated stop-early) and flags — while segments remain a derived view built on top.
    pub(crate) fn from_walk<T: but_core::RefMetadata>(
        repo: &gix::Repository,
        meta: &T,
        tip: gix::ObjectId,
        ref_name: Option<gix::refs::FullName>,
        project_meta: but_core::ref_metadata::ProjectMeta,
        options: crate::walk::Options,
        overlay: crate::walk::Overlay,
    ) -> anyhow::Result<Self> {
        let outcome = Self::from_walk_outcome(crate::Graph::walk_from_tip(
            repo,
            tip,
            ref_name.clone(),
            meta,
            project_meta.clone(),
            options,
            overlay,
        )?);
        Ok(outcome)
    }

    /// Like [`Self::from_walk`], but seeded from explicit `tips`.
    pub(crate) fn from_walk_seeds<T: but_core::RefMetadata>(
        repo: &gix::Repository,
        meta: &T,
        tips: Vec<crate::walk::Seed>,
        project_meta: but_core::ref_metadata::ProjectMeta,
        options: crate::walk::Options,
        overlay: crate::walk::Overlay,
    ) -> anyhow::Result<Self> {
        let mut outcome = Self::from_walk_outcome(crate::Graph::walk_from_seeds(
            repo,
            tips.clone(),
            meta,
            project_meta.clone(),
            options,
            overlay,
        )?);
        outcome.explicit_seeds = true;
        Ok(outcome)
    }

    /// Mark `id` as a GitButler-managed workspace commit when its message says so.
    pub(crate) fn mark_managed_ws_commit_by_message(
        &mut self,
        repo: &gix::Repository,
        id: gix::ObjectId,
    ) {
        if let Ok(commit) = repo.find_commit(id)
            && let Ok(message) = commit.message_raw()
            && crate::workspace::commit::is_managed_workspace_by_message(message)
        {
            self.managed_ws_commits.insert(id);
        }
    }

    /// Where traversal/HEAD started (a checkout inside a stack), if any. The projection forces a
    /// segment boundary here — there is always a segment starting at the entrypoint.
    pub fn entrypoint(&self) -> Option<gix::ObjectId> {
        self.entrypoint
    }

    /// The ref the entrypoint was checked out as, if any — it names the entrypoint segment.
    pub fn entrypoint_ref(&self) -> Option<&gix::refs::FullName> {
        self.entrypoint_ref.as_ref()
    }

    /// The ref placement table authored at build time, when this graph went through the builder.
    pub fn layout(&self) -> Option<&crate::ref_layout::RefLayout> {
        self.layout.as_ref()
    }

    /// Whether `id` is a GitButler-managed workspace commit (recognised by its message).
    pub(crate) fn is_managed_ws_commit(&self, id: gix::ObjectId) -> bool {
        self.managed_ws_commits.contains(&id)
    }

    /// Replace every node's attached refs with the CURRENT `refs_by_id` state — the
    /// write-through seam's enrichment refresh: an editor-mutated table still carries
    /// walk-time refs, but materialization has moved them. Matched entries are consumed.
    pub(crate) fn refresh_refs(
        &mut self,
        refs_by_id: &mut crate::walk::utils::RefsById,
        worktree_by_branch: &crate::walk::utils::WorktreeByBranch,
    ) {
        for (idx, node) in self.nodes.iter_mut().enumerate() {
            if self.tombstoned[idx] {
                node.commit.refs.clear();
                continue;
            }
            let id = node.commit.id;
            let mut refs: Vec<crate::RefInfo> = refs_by_id
                .remove(&id)
                .unwrap_or_default()
                .into_iter()
                .map(|rn| crate::RefInfo::from_ref(rn, id, worktree_by_branch))
                .collect();
            refs.sort_by(|a, b| a.ref_name.cmp(&b.ref_name));
            node.commit.refs = refs;
        }
        self.rebuild_by_ref();
    }

    /// Rebuild the ref-name lookup from the (live) nodes' attached refs.
    fn rebuild_by_ref(&mut self) {
        self.by_ref.clear();
        for (idx, node) in self.nodes.iter().enumerate() {
            if self.tombstoned[idx] {
                continue;
            }
            for r in &node.commit.refs {
                self.by_ref.insert(r.ref_name.clone(), idx);
            }
        }
    }

    /// Overwrite the node's flags — the write-through seam flags re-added tip regions
    /// Integrated, the walk's convention for target-seeded tips.
    pub(crate) fn set_flags(&mut self, idx: usize, flags: crate::CommitFlags) {
        self.nodes[idx].commit.flags = flags;
    }

    /// Set `flag` on exactly the (live) ancestors of `tips`, clearing it elsewhere — the
    /// write-through seam's flag refresh: an editor-mutated table carries walk-time flags
    /// (empty on editor-added nodes), while the rewalk derives each flag fresh. Which tips
    /// seed which flag is the walk's rule, spelled at the call sites.
    pub(crate) fn set_flag_on_ancestors(
        &mut self,
        flag: crate::CommitFlags,
        tips: impl IntoIterator<Item = gix::ObjectId>,
    ) {
        let mut marked: HashSet<gix::ObjectId> = HashSet::default();
        for tip in tips {
            if self.by_id.contains_key(&tip) && !marked.contains(&tip) {
                marked.extend(self.ancestor_set(tip));
            }
        }
        for (idx, node) in self.nodes.iter_mut().enumerate() {
            if self.tombstoned[idx] {
                continue;
            }
            node.commit
                .flags
                .set(flag, marked.contains(&node.commit.id));
        }
    }

    /// Bring a TOMBSTONED node holding `id` back to life — the write-through seam's tip
    /// revival: a stored/extra target the editor dropped from workspace history is still
    /// external context on disk, and the walk always seeds it as an integrated tip.
    ///
    /// Returns `true` if a tombstone actually came back to life — a revival flips the
    /// effective parents of every child that was tombstone-substituting through this node,
    /// so callers may need to re-validate them.
    pub(crate) fn revive(&mut self, id: gix::ObjectId) -> bool {
        if self.by_id.contains_key(&id) {
            return false;
        }
        let Some(idx) =
            (0..self.nodes.len()).find(|&i| self.tombstoned[i] && self.nodes[i].commit.id == id)
        else {
            return false;
        };
        self.tombstoned[idx] = false;
        self.by_id.insert(id, idx);
        true
    }

    /// The node holding `id`, if present.
    pub fn node(&self, id: gix::ObjectId) -> Option<&CommitNode> {
        self.by_id.get(&id).map(|&idx| &self.nodes[idx])
    }

    /// The node index of `id`, if present.
    pub fn index_of(&self, id: gix::ObjectId) -> Option<usize> {
        self.by_id.get(&id).copied()
    }

    /// Every commit id in the table, in node order.
    pub fn commit_ids(&self) -> impl Iterator<Item = gix::ObjectId> + '_ {
        self.nodes.iter().map(|n| n.commit.id)
    }

    /// The commit's CONNECTED parent list, first-parent first — parents the traversal severed
    /// (limits, integrated stop-early, display cuts) are omitted.
    ///
    /// An editor-dropped (TOMBSTONED) parent is substituted in place by its own parents,
    /// recursively — the same descent the editor's parent collection performs when it turns
    /// the structure into real commits — deduplicated along the descent, while plain
    /// duplicate slots are all kept (dup-parent workspace commits).
    pub(crate) fn all_parent_ids(&self, id: gix::ObjectId) -> Vec<gix::ObjectId> {
        let Some(&idx) = self.by_id.get(&id) else {
            return Vec::new();
        };
        // Fast path: no tombstoned parent to substitute through (always true for walk-built
        // tables), so the connected slots map straight to parent ids.
        if !self.has_tombstoned_parent(idx) {
            return self.nodes[idx]
                .commit
                .parent_ids
                .iter()
                .copied()
                .zip(&self.parent_slots[idx])
                .filter(|(_, slot)| slot.connected)
                .map(|(p, slot)| match slot.target_node_idx {
                    Some(t) => self.nodes[t].commit.id,
                    None => p,
                })
                .collect();
        }
        // The CONNECTED `(raw parent id, slot target)` pairs of `idx`, in slot order.
        let connected = |idx: usize| {
            self.nodes[idx]
                .commit
                .parent_ids
                .iter()
                .copied()
                .zip(&self.parent_slots[idx])
                .filter_map(|(p, slot)| slot.connected.then_some((p, slot.target_node_idx)))
                .collect::<Vec<_>>()
        };
        let mut potential: Vec<(gix::ObjectId, Option<usize>)> =
            connected(idx).into_iter().rev().collect();
        let mut seen_idx: std::collections::HashSet<usize> =
            potential.iter().filter_map(|(_, t)| *t).collect();
        let mut seen_raw: HashSet<gix::ObjectId> = potential
            .iter()
            .filter_map(|(p, t)| t.is_none().then_some(*p))
            .collect();
        let mut parents = Vec::new();
        while let Some((raw, target)) = potential.pop() {
            match target {
                Some(t) if self.tombstoned[t] => {
                    for (p_raw, p_target) in connected(t).into_iter().rev() {
                        let unseen = match p_target {
                            Some(pt) => seen_idx.insert(pt),
                            None => seen_raw.insert(p_raw),
                        };
                        if unseen {
                            potential.push((p_raw, p_target));
                        }
                    }
                }
                // A live target's CURRENT payload id is authoritative (raw agrees via
                // set_commit_id's child patching; this needs no such guarantee).
                Some(t) => parents.push(self.nodes[t].commit.id),
                None => parents.push(raw),
            }
        }
        parents
    }

    /// The RAW recorded parent ids of `idx` — the payload array, cut slots included. Unlike
    /// [`Self::all_parent_ids`] this does NOT substitute through tombstones, so a slot whose
    /// target was editor-dropped still shows the dropped commit's id.
    pub(crate) fn raw_parent_ids(&self, idx: usize) -> &[gix::ObjectId] {
        &self.nodes[idx].commit.parent_ids
    }

    /// `true` if any CONNECTED parent slot of `idx` targets a tombstone — traversal would
    /// substitute through it, so the raw recorded parents disagree with what a walk sees.
    pub(crate) fn has_tombstoned_parent(&self, idx: usize) -> bool {
        self.parent_slots[idx]
            .iter()
            .any(|slot| slot.connected && slot.target_node_idx.is_some_and(|t| self.tombstoned[t]))
    }

    /// All ancestors of `tip` (inclusive), following CONNECTED parent edges — history the
    /// traversal severed is not rejoined. Bounded by the table, which is the traversal-limited
    /// window, not the repository.
    pub fn ancestor_set(&self, tip: gix::ObjectId) -> HashSet<gix::ObjectId> {
        let mut set = HashSet::default();
        let mut queue = std::collections::VecDeque::from([tip]);
        while let Some(c) = queue.pop_front() {
            if set.insert(c) {
                queue.extend(self.all_parent_ids(c));
            }
        }
        set
    }

    /// The first commit along `tip`'s first-parent spine for which `is_in_set` holds — where an
    /// outside line (a remote, the target, an outside checkout, a branch that advanced past the
    /// workspace) rejoins a marked region of the graph.
    pub(crate) fn first_on_spine(
        &self,
        tip: gix::ObjectId,
        is_in_set: impl Fn(usize) -> bool,
    ) -> Option<gix::ObjectId> {
        let mut cursor = self.index_of(tip);
        while let Some(c) = cursor {
            if is_in_set(c) {
                return Some(self.id_at(c));
            }
            cursor = self.first_parent_at(c);
        }
        None
    }

    /// The nearest commit common to `target` and EVERY parent of `merge_commit` — the base all
    /// of the merge's parent lines and the target converge on. BFS from `merge_commit` over all
    /// parents, so the nearest such commit wins.
    pub(crate) fn lowest_common_base(
        &self,
        merge_commit: gix::ObjectId,
        target: gix::ObjectId,
    ) -> Option<gix::ObjectId> {
        let mut common = self.ancestor_set(target);
        for parent in self.all_parent_ids(merge_commit) {
            let parent_ancestors = self.ancestor_set(parent);
            common.retain(|c| parent_ancestors.contains(c));
        }
        let mut seen = HashSet::default();
        let mut queue = std::collections::VecDeque::from([merge_commit]);
        while let Some(c) = queue.pop_front() {
            if common.contains(&c) {
                return Some(c);
            }
            if seen.insert(c) {
                queue.extend(self.all_parent_ids(c));
            }
        }
        None
    }

    /// [`Self::ancestor_set`] in handle space: `marks[idx]` for every reachable LIVE node. The
    /// walk passes THROUGH tombstones without marking them, matching the id-space substitution.
    /// Query membership via [`Self::index_of`].
    pub fn ancestor_marks(&self, tip: gix::ObjectId) -> Vec<bool> {
        let mut marks = vec![false; self.nodes.len()];
        let mut queue: Vec<usize> = self.index_of(tip).into_iter().collect();
        while let Some(c) = queue.pop() {
            if std::mem::replace(&mut marks[c], true) {
                continue;
            }
            for slot in &self.parent_slots[c] {
                if slot.connected
                    && let Some(p) = slot.target_node_idx
                {
                    queue.push(p);
                }
            }
        }
        for (i, dead) in self.tombstoned.iter().enumerate() {
            if *dead {
                marks[i] = false;
            }
        }
        marks
    }

    /// Return `true` if any of `id`'s recorded parents is not CONNECTED in this table — the
    /// traversal cut history here (limits, integrated stop-early), so ancestry continues
    /// beyond what the table can see.
    pub fn has_cut_parents(&self, id: gix::ObjectId) -> bool {
        self.slots_of(id)
            .is_some_and(|slots| slots.iter().any(|slot| !slot.connected))
    }

    /// [`Self::has_cut_parents`] by handle.
    pub fn has_cut_parents_at(&self, idx: usize) -> bool {
        self.parent_slots[idx].iter().any(|slot| !slot.connected)
    }

    /// The commit that `ref_name` points at, if present in the table.
    pub fn commit_by_ref(&self, ref_name: &gix::refs::FullNameRef) -> Option<gix::ObjectId> {
        self.by_ref
            .get(ref_name)
            .filter(|&&idx| !self.tombstoned[idx])
            .map(|&idx| self.nodes[idx].commit.id)
    }

    /// The reference names pointing at `id`.
    pub(crate) fn refs_at(&self, id: gix::ObjectId) -> Vec<gix::refs::FullName> {
        self.node(id)
            .map(|n| n.commit.refs.iter().map(|r| r.ref_name.clone()).collect())
            .unwrap_or_default()
    }

    /// The parents of `id` that are present in this table, first-parent first.
    pub fn parents(&self, id: gix::ObjectId) -> impl Iterator<Item = gix::ObjectId> + '_ {
        self.slots_of(id)
            .unwrap_or_default()
            .iter()
            .filter_map(|slot| slot.target_node_idx.map(|idx| self.nodes[idx].commit.id))
    }

    /// The first parent of `id` (the next commit walking down first-parent), if present.
    pub fn first_parent(&self, id: gix::ObjectId) -> Option<gix::ObjectId> {
        let slot = self.slots_of(id)?.first()?;
        let target = slot.target_node_idx.filter(|_| slot.connected)?;
        Some(self.nodes[target].commit.id)
    }

    // --- HANDLE-based reads: the builder's hot loops speak node indices (no `by_id` hashing, no
    // per-call allocation). No tombstone substitution — the builder only sees walk-built or
    // compacted tables (the write-through seam compacts before building).

    /// The id of the (live) node at `idx`.
    pub(crate) fn id_at(&self, idx: usize) -> gix::ObjectId {
        self.nodes[idx].commit.id
    }

    /// The node at `idx`.
    pub(crate) fn node_at(&self, idx: usize) -> &CommitNode {
        &self.nodes[idx]
    }

    /// The CONNECTED, PRESENT parents of `idx` in slot order — the handle-space read behind
    /// [`Self::all_parent_ids`]'s fast path.
    pub(crate) fn connected_parents_at(&self, idx: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(
            !self.has_tombstoned_parent(idx),
            "no substitution by handle"
        );
        self.parent_slots[idx]
            .iter()
            .filter(|slot| slot.connected)
            .filter_map(|slot| slot.target_node_idx)
    }

    /// `all_parent_ids(id_at(idx)).len()` without materializing the list — connected slots,
    /// absent targets included.
    pub(crate) fn connected_parent_count_at(&self, idx: usize) -> usize {
        debug_assert!(
            !self.has_tombstoned_parent(idx),
            "no substitution by handle"
        );
        self.parent_slots[idx]
            .iter()
            .filter(|slot| slot.connected)
            .count()
    }

    /// [`Self::first_parent`] by handle.
    pub(crate) fn first_parent_at(&self, idx: usize) -> Option<usize> {
        let slot = self.parent_slots[idx].first()?;
        slot.target_node_idx.filter(|_| slot.connected)
    }

    /// [`Self::children`] by handle.
    pub(crate) fn children_at(&self, idx: usize) -> &[usize] {
        &self.children[idx]
    }

    /// The children of `id` (commits that list `id` as a parent). More than one means a branch point.
    pub fn children(&self, id: gix::ObjectId) -> impl Iterator<Item = gix::ObjectId> + '_ {
        self.by_id
            .get(&id)
            .into_iter()
            .flat_map(move |&idx| self.children[idx].iter().map(|&c| self.nodes[c].commit.id))
    }

    /// Recompute `generation` for every node (longest path from a root, by Kahn order). Cheap; the
    /// table is small.
    pub(crate) fn recompute_generations(&mut self) {
        // Process in topological order (parents before children) so a child's generation is the max
        // over its present parents + 1.
        let order = self.toposort_parents_first();
        for idx in order {
            let generation = self.parent_slots[idx]
                .iter()
                .filter_map(|slot| slot.target_node_idx)
                .map(|pidx| self.nodes[pidx].generation + 1)
                .max()
                .unwrap_or(0);
            self.nodes[idx].generation = generation;
        }
    }

    /// Topological order with parents before children (history order), over PRESENT slots —
    /// connectivity ignored, like the generation formula. Propagating via the connected-only
    /// `children` adjacency would strand nodes behind severed edges.
    fn toposort_parents_first(&self) -> Vec<usize> {
        let mut children = vec![Vec::new(); self.nodes.len()];
        let mut indegree = vec![0usize; self.nodes.len()];
        for (idx, slots) in self.parent_slots.iter().enumerate() {
            for slot in slots {
                if let Some(pidx) = slot.target_node_idx {
                    children[pidx].push(idx);
                    indegree[idx] += 1;
                }
            }
        }
        let mut queue: std::collections::VecDeque<usize> = (0..self.nodes.len())
            .filter(|&i| indegree[i] == 0)
            .collect();
        let mut out = Vec::with_capacity(self.nodes.len());
        while let Some(idx) = queue.pop_front() {
            out.push(idx);
            for &child in &children[idx] {
                indegree[child] -= 1;
                if indegree[child] == 0 {
                    queue.push_back(child);
                }
            }
        }
        out
    }

    // --- The EDITOR MUTATION SURFACE ---
    //
    // but-rebase's editor mutates a CommitGraph in place: node indices are the stable
    // ids, `ObjectId` is payload. These writes maintain `by_id`, the children adjacency, and
    // the raw `parent_ids` payload. `generation` is creation-time data and NOT maintained.

    /// Append a fresh node with `id` as payload (`None` = born tombstoned) and no parents.
    pub fn add_node(&mut self, id: Option<gix::ObjectId>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(CommitNode {
            commit: Commit {
                id: id.unwrap_or_else(|| gix::ObjectId::null(gix::hash::Kind::Sha1)),
                parent_ids: Vec::new(),
                flags: CommitFlags::empty(),
                refs: Vec::new(),
            },
            generation: 0,
        });
        self.parent_slots.push(Vec::new());
        self.children.push(Vec::new());
        self.tombstoned.push(id.is_none());
        if let Some(id) = id {
            self.by_id.insert(id, idx);
        }
        idx
    }

    /// Overwrite the node's payload id — `None` tombstones it in place (the stale payload is
    /// retained, id-based lookups stop finding it), `Some` (re)vitalizes it.
    pub fn set_node_id(&mut self, idx: usize, id: Option<gix::ObjectId>) {
        match id {
            Some(id) => {
                self.tombstoned[idx] = false;
                self.set_commit_id(idx, id);
            }
            None => {
                let old = self.nodes[idx].commit.id;
                if self.by_id.get(&old) == Some(&idx) {
                    self.by_id.remove(&old);
                }
                self.tombstoned[idx] = true;
            }
        }
    }

    /// Rewrite the commit id at `idx` IN PLACE — THE rebase write. The node index, its slots,
    /// and its children survive; `by_id`, the children's raw `parent_ids` entries, and the
    /// id-addressed markers (entrypoint, managed-ws) follow the payload.
    pub fn set_commit_id(&mut self, idx: usize, id: gix::ObjectId) {
        let old = self.nodes[idx].commit.id;
        self.nodes[idx].commit.id = id;
        if self.by_id.get(&old) == Some(&idx) {
            self.by_id.remove(&old);
        }
        self.by_id.insert(id, idx);
        for child in self.children[idx].clone() {
            for (slot_pos, slot) in self.parent_slots[child].iter().enumerate() {
                if slot.target_node_idx == Some(idx) {
                    self.nodes[child].commit.parent_ids[slot_pos] = id;
                }
            }
        }
        if self.entrypoint == Some(old) {
            self.entrypoint = Some(id);
        }
        if self.managed_ws_commits.remove(&old) {
            self.managed_ws_commits.insert(id);
        }
    }

    /// Overwrite `idx`'s whole parent array — the editor's ordered-slot write. Every new slot
    /// is PRESENT and CONNECTED; the raw `parent_ids` payload derives from the targets and the
    /// children adjacency follows.
    pub fn set_parents(&mut self, idx: usize, parents: Vec<usize>) {
        for slot in std::mem::take(&mut self.parent_slots[idx]) {
            if let Some(t) = slot.target_node_idx
                && let Some(pos) = self.children[t].iter().position(|&c| c == idx)
            {
                self.children[t].remove(pos);
            }
        }
        self.nodes[idx].commit.parent_ids =
            parents.iter().map(|&p| self.nodes[p].commit.id).collect();
        self.parent_slots[idx] = parents
            .iter()
            .map(|&p| ParentSlot {
                target_node_idx: Some(p),
                connected: true,
            })
            .collect();
        for &p in &parents {
            self.children[p].push(idx);
        }
    }

    /// Reconcile every live node's parents with the odb — the write-through seam's edge refresh.
    /// After materialization the odb is authoritative for every kept id: an editor pick applied
    /// AS-IS carries no arena edges at all (parents implied by the odb), and a `preserved_parents`
    /// pick was WRITTEN with parents its arena position never had (edit mode's parent override).
    /// A node is kept only when its RAW recorded parents equal its odb parents AND no connected
    /// slot targets a tombstone — the projection reads the raw payload, so a stale id left behind
    /// by an editor drop is as much a divergence as a wrong edge. Walk cuts (absent slots with
    /// odb-true payload) survive. Anything else is rewired to the odb parents, adding (or
    /// reviving) missing commits recursively, each reconciled the same way.
    pub(crate) fn complete_parents_from_odb(
        &mut self,
        repo: &crate::walk::overlay::OverlayRepo<'_>,
    ) -> anyhow::Result<()> {
        let mut queue: Vec<gix::ObjectId> = (0..self.node_count())
            .filter_map(|idx| self.node_payload(idx))
            .collect();
        while let Some(id) = queue.pop() {
            let idx = self.index_of(id).expect("queued ids are live");
            let Ok(commit) = repo.find_commit(id) else {
                continue;
            };
            let mut odb_parents: Vec<_> = commit.parent_ids().map(|p| p.detach()).collect();
            // Collapse exact duplicate parents like the walk and the graph reader do (a
            // workspace merge encodes empty stacks as repeated parents).
            if odb_parents.len() > 1 {
                let mut deduped = Vec::with_capacity(odb_parents.len());
                for p in odb_parents {
                    if !deduped.contains(&p) {
                        deduped.push(p);
                    }
                }
                odb_parents = deduped;
            }
            if self.raw_parent_ids(idx) == odb_parents && !self.has_tombstoned_parent(idx) {
                continue;
            }
            let mut revived = Vec::new();
            let parent_indices = odb_parents
                .iter()
                .map(|&p| {
                    if self.revive(p) {
                        revived.push(p);
                    }
                    self.index_of(p).unwrap_or_else(|| {
                        queue.push(p);
                        self.add_node(Some(p))
                    })
                })
                .collect();
            self.set_parents(idx, parent_indices);
            // A revived node is newly live — it wasn't in the initial sweep, so validate it now.
            // (Its children need no re-queue: any child kept earlier already passed the
            // tombstone-free check, so it can't have been substituting through this node.)
            queue.extend(revived);
        }
        Ok(())
    }

    /// Clear the walk's GOAL bits (the bits beyond [`CommitFlags::all()`](crate::CommitFlags))
    /// from every node. Goal numbering is traversal-order ephemera — a node that survived a
    /// rewrite carries whatever bits the ORIGINAL walk assigned, which a fresh walk would
    /// number differently — and nothing after the walk reads goals.
    pub(crate) fn strip_goal_flags(&mut self) {
        for node in &mut self.nodes {
            node.commit.flags &= crate::CommitFlags::all();
        }
    }

    /// Drop tombstoned nodes and reindex, leaving a table indistinguishable from one built
    /// without them — what the write-through seam hands to projection, so a carried table
    /// never leaks editor tombstones to the next consumer. The caller must first ensure no
    /// live slot targets a tombstone (the seam's odb reconciliation guarantees it); such a
    /// slot would degrade to "parent outside the graph" here.
    pub fn compact(&mut self) {
        if !self.tombstoned.iter().any(|&t| t) {
            return;
        }
        let mut remap: Vec<Option<usize>> = vec![None; self.nodes.len()];
        let mut next = 0;
        for (idx, remapped) in remap.iter_mut().enumerate() {
            if !self.tombstoned[idx] {
                *remapped = Some(next);
                next += 1;
            }
        }
        let mut nodes = Vec::with_capacity(next);
        let mut parent_slots = Vec::with_capacity(next);
        for idx in 0..self.nodes.len() {
            if remap[idx].is_none() {
                continue;
            }
            nodes.push(self.nodes[idx].clone());
            parent_slots.push(
                self.parent_slots[idx]
                    .iter()
                    .map(|slot| ParentSlot {
                        target_node_idx: slot.target_node_idx.and_then(|t| remap[t]),
                        connected: slot.connected,
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let by_id: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(idx, n): (_, &CommitNode)| (n.commit.id, idx))
            .collect();
        let mut children = vec![Vec::new(); nodes.len()];
        for (idx, slots) in parent_slots.iter().enumerate() {
            for slot in slots {
                if let Some(pidx) = slot.target_node_idx {
                    children[pidx].push(idx);
                }
            }
        }
        self.managed_ws_commits.retain(|id| by_id.contains_key(id));
        self.nodes = nodes;
        self.by_id = by_id;
        self.parent_slots = parent_slots;
        self.children = children;
        self.tombstoned = vec![false; next];
        self.rebuild_by_ref();
    }

    /// Arena length, tombstones included.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The payload id at `idx` — `None` for tombstones.
    pub fn node_payload(&self, idx: usize) -> Option<gix::ObjectId> {
        (!self.tombstoned[idx]).then(|| self.nodes[idx].commit.id)
    }

    /// The parent TARGETS of `idx` in slot order. Only meaningful once every slot is
    /// editor-authored (present) — walk-built tables can have absent slots.
    pub fn parent_indices(&self, idx: usize) -> Vec<usize> {
        self.parent_slots[idx]
            .iter()
            .map(|slot| {
                slot.target_node_idx
                    .expect("editor-authored slots are present")
            })
            .collect()
    }

    /// The PRESENT parent targets of `idx` in slot order — absent (walk-cut) slots are
    /// skipped, unlike [`Self::parent_indices`] which requires every slot to be present.
    pub fn present_parent_indices(&self, idx: usize) -> Vec<usize> {
        self.parent_slots[idx]
            .iter()
            .filter_map(|slot| slot.target_node_idx)
            .collect()
    }

    /// Build from a set of commits, every raw edge connected. Commits whose parents are
    /// outside the set are simply roots of this subgraph (a partial graph), mirroring how the
    /// rebase editor handles missing parents via `preserved_parents`.
    #[cfg(test)]
    fn from_commits(
        commits: impl IntoIterator<Item = Commit>,
        entrypoint: Option<gix::ObjectId>,
    ) -> Self {
        let nodes: Vec<CommitNode> = commits
            .into_iter()
            .map(|commit| CommitNode {
                commit,
                generation: 0,
            })
            .collect();
        let by_id: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(idx, n)| (n.commit.id, idx))
            .collect();

        // The handle-based edges: one slot per raw parent entry, presence from the index, every
        // edge connected until a walk restricts it. Reverse adjacency mirrors the present slots.
        let parent_slots: Vec<Vec<ParentSlot>> = nodes
            .iter()
            .map(|n| {
                n.commit
                    .parent_ids
                    .iter()
                    .map(|parent| ParentSlot {
                        target_node_idx: by_id.get(parent).copied(),
                        connected: true,
                    })
                    .collect()
            })
            .collect();
        let mut children = vec![Vec::new(); nodes.len()];
        for (idx, slots) in parent_slots.iter().enumerate() {
            for slot in slots {
                if let Some(pidx) = slot.target_node_idx {
                    children[pidx].push(idx);
                }
            }
        }

        let tombstoned = vec![false; nodes.len()];
        let mut table = CommitGraph {
            nodes,
            by_id,
            by_ref: Default::default(),
            parent_slots,
            children,
            tombstoned,
            entrypoint,
            entrypoint_ref: None,
            managed_ws_commits: HashSet::default(),
            hard_limit_hit: false,
            seeds: Vec::new(),
            explicit_seeds: false,
            layout: None,
        };
        table.recompute_generations();
        table.rebuild_by_ref();
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CommitFlags;

    fn id(b: u8) -> gix::ObjectId {
        let mut bytes = [0u8; 20];
        bytes[0] = b;
        gix::ObjectId::from_bytes_or_panic(&bytes)
    }

    fn commit(b: u8, parents: &[u8]) -> Commit {
        Commit {
            id: id(b),
            parent_ids: parents.iter().map(|&p| id(p)).collect(),
            flags: CommitFlags::empty(),
            refs: Vec::new(),
        }
    }

    #[test]
    fn children_generation_and_first_parent_walk() {
        // Linear: 3 -> 2 -> 1 (child -> parent).
        let g = CommitGraph::from_commits(
            [commit(3, &[2]), commit(2, &[1]), commit(1, &[])],
            Some(id(3)),
        );
        assert_eq!(g.first_parent(id(3)), Some(id(2)));
        assert_eq!(g.first_parent(id(1)), None);
        assert_eq!(g.children(id(1)).collect::<Vec<_>>(), vec![id(2)]);
        // Generation increases with history depth.
        assert_eq!(g.node(id(1)).unwrap().generation, 0);
        assert_eq!(g.node(id(3)).unwrap().generation, 2);
    }
}
