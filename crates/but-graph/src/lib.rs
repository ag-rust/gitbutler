//! A graph data structure for seeing the Git commit graph as segments.
//!
//! ### The pipeline
//!
//! Everything in this crate flows through one pipeline with a single substrate, the
//! [`CommitGraph`]:
//!
//! ```text
//! repository ──walk──▶ CommitGraph ──build──▶ Graph (segments) ──project──▶ Workspace
//!                          ▲                                                     │
//!                          └───────────── but-rebase edits ◀────────────────────┘
//! ```
//!
//! * **walk** ([`walk`], entered through [`Graph::from_head`] and [`Graph::from_tip`]): the
//!   traversal seeds from `HEAD`, the workspace ref, the target, and the stack branches; obeys
//!   goals and limits; propagates flags; and accumulates the [`CommitGraph`] — an arena of
//!   commits with ordered parent arrays, the connectivity the traversal actually followed, and
//!   every encountered ref attached as data.
//! * **build** (the private `build` module, entered through [`graph_from_repository`]):
//!   gather-then-build. Pure fact and planning passes decide the full segment structure as
//!   data — boundaries, chains from workspace metadata, the name lifecycle, the stored
//!   [`ref_layout`] — then materialization mints the segment [`Graph`] in its final shape.
//!   Segments are FINAL at creation: commits, names, connections and endpoints never change
//!   afterwards; only link-level fields (sibling and remote-tracking links) are filled in by
//!   later passes.
//! * **project** ([`workspace`], entered through [`Graph::into_workspace`]): the application
//!   view — stacks of first-parent chains of segments, with integration status against the
//!   target. Projections are lossy; they carry links back to the segments they came from.
//! * **edit** (`but-rebase`): the editor is created FROM the [`CommitGraph`] (mutability follows
//!   reachability), and an edited commit graph re-enters the same build via
//!   [`workspace_from_commit_graph`] — edit previews and fresh walks share one code path.
//!   `but-workspace` sits on top of both as the operations layer.
//!
//! The segment view exists because stacks alone degenerate information: a first-parent walk to
//! the merge-base flattens exactly the merges the application helps create. The [`Graph`] keeps
//! that shape for display — segments own their outgoing [`Connection`]s, and a stack is merely
//! a projection of it — while the [`CommitGraph`] underneath stays the record everything is
//! derived from and every edit is applied to.
//!
//! ### New Workspace Concepts
//!
//! The workspace is merely a projection of *The Graph*, and as such is mostly useful for display and user interaction.
//! In the end it boils down to passing commit-hashes, or plain-`usize` segment ids at most.
//!
//! The workspace has been redesigned from the ground up for flexibility, enabling new user-experiences. To help thinking
//! about these, a few new concepts will be good to know about.
//!
//! #### Entrypoint
//!
//! *The Graph* knows where its traversal started as *Entrypoint*, even though it may extend beyond the entrypoint as it
//! needs to discover possible surrounding workspaces and the target branches that come with them.
//! In practice, the entrypoint relates to the position of the Git `HEAD` reference, and with that it relates to what
//! the user currently sees in their worktree.
//!
//! #### Early End of Traversal
//!
//! During traversal there are mandatory goals, but when reached the traversal usually obeys a limit, if configured.
//! This is particularly relevant in open-ended traversals outside of workspaces, they can go on until the end of history,
//! literally.
//!
//! For that reason, whenever a commit isn't the end of the graph, but the end traversal as a [limit was hit](walk::Options::with_limit_hint),
//! it will be flagged as such.
//!
//! This way one can visualize such Early Ends, and allow the user to extend the traversal selectively the next time it
//! is performed.
//!
//! Despite that, one has to learn how to deal with possible huge graphs, and possible workspaces with a lot of commits,
//! and [a hard limit](walk::Options::with_hard_limit()) as long as downstream cannot deal with this on their own.
//!
//! #### Managed Workspaces, and unmanaged ones
//!
//! A Workspace is considered managed if it [has workspace metadata](Workspace::metadata). This is typically
//! only the case for workspaces that have been created by GitButler.
//!
//! Workspaces without such metadata can be anything, and are usually just made up to allow GitButler to work with it based
//! on any `HEAD` position. These should be treated with care, and multi-lane workflows should generally be avoided - these
//! are reserved to managed Workspaces with the managed merge commit that comes with them.
//!
//! #### Optional Targets
//!
//! Even on *Managed Workspaces*, target references are now optional. This makes it possible to have a workspace that doesn't
//! know if it's integrated or not. These are the reason a [soft limit](walk::Options::with_limit_hint()) must always be set
//! to assure the traversal doesn't fetch the entire Git history.
//!
//! This, however, also means that the workspace creation doesn't have to be interrupted by a "what's your target" prompt anymore.
//! Instead, this can be prompted once an action first requires it.
//!
//! #### Commit Flags and Segment Flags
//!
//! For convenience, various boolean parameters have been aggregated into [bitflags](Commit::flags). Thanks to the way *The Graph*
//! is traversed, we know that the first commit of any [graph segment](Segment) will always bear the flags that are also used by every other commit
//! contained within it. Thus, a segment's flags (`Segment::non_empty_flags_of_first_commit()`) are equivalent to the flags of
//! their first commit.
//!
//! The same is *not* true for [stack segments](workspace::StackSegment), i.e. segments within a [workspace projection](Workspace).
//! The reason for this is that they are first-parent aggregations of one *or more* [graph segments](Segment), and thus have multiple
//! sets of flags, possibly one per [segment](Segment).
//!
//! #### The 'frozen' Commit-Flag
//!
//! [`CommitFlags::NotInRemote`] marks commits NOT reachable from any remote-seeded tip. The
//! workspace projection inverts it into
//! [`StackCommitFlags::ReachableByRemote`](workspace::StackCommitFlags): commits others may
//! already have observed, to be treated as frozen and not manipulated casually.
//!
//! ### Build decisions
//!
//! #### Commits are owned by Segments
//!
//! A commit can only be owned by a single segment. Thus, there are empty *named* segments which point at other segments,
//! effectively representing a reference.
//! Which of these references gets to own a commit is a *planning* decision.
//!
//! #### Planning chains from metadata
//!
//! *The Graph* is created from traversing the Git commit graph. Thus, information that is not contained in it,
//! like workspace metadata, has to shape the segmented graph as it is built.
//!
//! That way, we can create *stacks* as independent branches and dependent branches inside of them without having
//! a single commit to differentiate their respective branches from each other.
//!
//! Imagine a repository with a single commit `73a30f8` with the following Git references pointing to it: `gitbutler/workspace`,
//! `stack1-segment1`, `stack1-segment2`, `stack2-segment1`, and `refs/remotes/origin/main`.
//!
//! A naive segmentation of the traversal would look like this:
//!
//! ```text
//!   ┌────────────────────┐
//!   │    origin/main     │
//!   └────────────────────┘
//!              │
//!              ▼
//! ┌────────────────────────┐
//! │gitbutler/workspace     │
//! │------------------------│
//! │73a30f8 ►stack1-segment1│
//! │        ►stack1-segment2│
//! │        ►stack2-segment1│
//! │        ►main           │
//! └────────────────────────┘
//! ```
//!
//! This is because `gitbutler/workspace` owns `73a30f8`, with `origin/main` merely pointing to
//! that commit; the other references would be plain refs on it.
//!
//! The chain plan instead reads [workspace metadata](but_core::ref_metadata::Workspace::stacks) before any
//! segment exists and decides which refs form chains of empty segments. Materialization then mints this
//! shape directly:
//!
//! ```text
//! ┌───────────────────┐
//! │    origin/main    │
//! └───────────────────┘
//!            │            ┌────────────────────┐
//!            │            │gitbutler/workspace │
//!            │            └────────────────────┘
//!            │                       │
//!            │             ┌─────────┴─────────┐
//!            │             │                   │
//!            │             ▼                   │
//!            │     ┌───────────────┐           │
//!            │     │stack1-segment1│           ▼
//!            │     └───────────────┘   ┌───────────────┐
//!            │             │           │stack2-segment1│
//!            │             ▼           └───────────────┘
//!            │     ┌───────────────┐           │
//!            │     │stack1-segment2│           │
//!            │     └───────────────┘           │
//!            │             │                   │
//!            │             └─────────┬─────────┘
//!            │                       │
//!            │                       ▼
//!            │                  ┌────────┐
//!            │                  │  main  │
//!            └─────────────────▶│ ------ │
//!                               │ 73a30f │
//!                               └────────┘
//! ```
//!
//! #### Projection
//!
//! A projection is a mapping of the segmented graph to any shape an application needs, and for any purpose.
//! Projections are inherently lossy, so they carry links back to the segments the information was
//! extracted from — and manipulation never operates on a projection: the rebase editor edits the
//! [`CommitGraph`], and re-building from the edited substrate yields the next segment graph and
//! its projections.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod segment;
pub use segment::{
    Commit, CommitFlags, RefInfo, Segment, SegmentFlags, SegmentMetadata, StopCondition, Worktree,
    WorktreeKind,
};

mod api;
pub use api::FirstParent;
/// Produce a graph from a Git repository.
pub mod walk;

#[path = "projection/mod.rs"]
pub mod workspace;
pub use workspace::workspace::Workspace;

mod utils;

mod segment_graph;
/// The segment graph, where segments directly own their outgoing connections.
pub use segment_graph::{Connection, Direction};

/// The commit-first graph flattened out of the raw traversal — the substrate every graph build
/// starts from. See the module docs.
mod commit_graph;
pub use commit_graph::{CommitGraph, CommitNode};
/// The graph builders: derive the segment [`Graph`] from a [`CommitGraph`]. See the module docs.
mod build;
/// The metadata-driven ref placement table stored on the commit graph. See the module docs.
pub mod ref_layout;
pub use build::{graph_from_repository, workspace_from_commit_graph, workspace_from_repository};
pub(crate) use build::{graph_from_repository_seeds, graph_from_repository_unmanaged};

mod statistics;
pub use statistics::Statistics;

mod debug;

/// Edges to other segments are the index into the list of local commits of the parent segment.
/// That way we can tell where a segment branches off, despite the graph only connecting segments, and not commits.
pub type CommitIndex = usize;

/// A graph of connected segments that represent a section of the actual commit-graph.
#[derive(Default, Clone)]
#[must_use]
pub struct Graph {
    /// The segment arena: index = segment id. Removing a segment tombstones its slot, which is
    /// reused (LIFO) before a fresh id is handed out, so segment ids stay stable across
    /// construction-time surgery.
    pub(crate) segments: Vec<Option<Segment>>,
    /// The maintained reverse index: index = target segment; one source entry per connection.
    /// Kept in sync by the edge methods so per-segment incoming queries are O(degree).
    pub(crate) incoming: Vec<Vec<usize>>,
    /// Tombstoned arena slots, reused LIFO.
    pub(crate) free: Vec<usize>,
    /// From where the graph was created. This is useful if one wants to focus on a subset of the graph.
    ///
    /// This is `None` only for a freshly default-initialized graph, or while a graph is being assembled before
    /// the first segment is inserted. Graphs returned by the traversal constructors are expected to have an
    /// entrypoint, even for unborn refs; in that case the segment is present and the commit is
    /// [`EntryPointCommit::Unborn`].
    ///
    /// The second value is the traversal tip, if there is one.
    ///
    /// Post-processing may move the entrypoint to an empty synthetic segment, for example a named workspace ref
    /// without a workspace commit. The segment still marks the ref's place in the graph, but it has no local
    /// commit slot that can hold the ref target. In that case the commit id is kept here so
    /// [`Graph::redo_traversal()`] can keep using the original traversal seed if the ref can no
    /// longer be resolved in the overlay/repository.
    entrypoint: Option<(usize, EntryPointCommit)>,
    /// The ref_name used when starting the graph traversal. It is set to help assure that the entrypoint stays
    /// on the correctly named segment, as chain creation can splice empty segments for independent and
    /// dependent branches around the entrypoint's position.
    entrypoint_ref: Option<gix::refs::FullName>,
    /// The validated snapshot of [`branch_stack_order`](but_core::RefMetadata::branch_stack_order)
    /// for the entrypoint ref: the tip-to-base branch chains GitButler created in
    /// ad-hoc/single-branch mode, filtered at build time to refs that still exist.
    ad_hoc_branch_stack_orders: Vec<Vec<gix::refs::FullName>>,
    /// The options used to create the graph, which allows it to regenerate itself after something
    /// possibly changed. This can also be used to simulate changes by injecting would-be information.
    /// Public to be able to change it before calling [Graph::redo_traversal()].
    pub options: walk::Options,
    /// Project-wide metadata used for target ref, target commit, and push remote resolution.
    pub project_meta: but_core::ref_metadata::ProjectMeta,
    /// All remote names that aren't URLs and that were retrieved during the traversal.
    ///
    /// They are useful to extract remote names from remote tracking refs like `refs/remotes/origin/master`,
    /// which may have slashes in them.
    pub symbolic_remote_names: Vec<String>,
    /// The local → remote tracking-branch relationships the builder derived from the repository
    /// (git-configured bindings plus name-deduction against the workspace's symbolic remotes).
    /// Carried so consumers resolve tracking relationships as data instead of navigating
    /// segment links. Empty for graphs not born from the CommitGraph builders.
    pub(crate) remote_tracking: std::collections::HashMap<gix::refs::FullName, gix::refs::FullName>,
    /// The commit graph this segment graph was assembled from — the commit-addressed substrate
    /// consumers migrate to as the segment view winds down. It is also the record for traversal
    /// carries (hard-limit signal, seed tips). Empty (default) only for graphs not born from
    /// the CommitGraph builders (defaults, hand-assembled test graphs).
    pub(crate) commit_graph: CommitGraph,
}

/// Like the derived implementation, but omitting the carried [`CommitGraph`] (the debug dump
/// documents the segment view, and the substrate has its own renderers) and the `incoming`
/// index (derived from the connections and only noise in snapshots).
impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("segments", &self.segments)
            .field("free", &self.free)
            .field("entrypoint", &self.entrypoint)
            .field("entrypoint_ref", &self.entrypoint_ref)
            .field("seeds", &self.seeds())
            .field(
                "ad_hoc_branch_stack_orders",
                &self.ad_hoc_branch_stack_orders,
            )
            .field("hard_limit_hit", &self.hard_limit_hit())
            .field("options", &self.options)
            .field("project_meta", &self.project_meta)
            .field("symbolic_remote_names", &self.symbolic_remote_names)
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum EntryPointCommit {
    /// The traversal seed is known.
    ///
    /// This is an object id rather than a synthetic index because an empty entrypoint segment has no commit slot
    /// to index into, nor is it reliably reachable by traversing the graph. If the commit is present in the
    /// entrypoint segment, its [`CommitIndex`] can be derived from the segment and this id.
    /// This happens when a workspace reference doesn't have an (unneeded) workspace merge commit,
    /// and is connected to one or more named empty segments which themselves only point to the workspace base.
    /// Then from Git's point of view, all involved refs point to the workspace base, but for use it's
    /// already a graph, and one that isn't representable even with symbolic refs as the workspace ref segment
    /// can easily point to multiple refs at the same time.
    AtCommit(gix::ObjectId),
    /// The traversal started from an unborn ref and has no tip commit.
    Unborn,
}

impl EntryPointCommit {
    fn index_in(self, segment: &Segment) -> Option<CommitIndex> {
        match self {
            EntryPointCommit::AtCommit(id) => segment.commit_index_of(id),
            EntryPointCommit::Unborn => None,
        }
    }

    pub(crate) fn object_id(self) -> Option<gix::ObjectId> {
        match self {
            EntryPointCommit::AtCommit(id) => Some(id),
            EntryPointCommit::Unborn => None,
        }
    }
}

impl Graph {
    /// Return the entrypoint as a segment plus the optional position of its tip commit in that segment.
    ///
    /// The graph stores the tip as an object id so it stays valid when chain creation places the entrypoint
    /// on an empty or otherwise synthetic segment. Some graph-local consumers, like debug rendering and
    /// statistics, still need the segment-relative commit position to mark where the entrypoint appears in
    /// the current graph shape, so it is derived here instead of being stored as mutable state.
    fn entrypoint_location(&self) -> Option<(usize, Option<CommitIndex>)> {
        let (segment_index, commit) = self.entrypoint?;
        let segment = self.segment(segment_index)?;
        Some((segment_index, commit.index_in(segment)))
    }
}

/// A resolved entry point into the graph for easy access to the entrypoint segment and,
/// if present, the commit that started traversal along with the segment that owns it.
#[derive(Debug, Copy, Clone)]
pub struct EntryPoint<'graph> {
    /// The segment that served starting point for the traversal into this graph.
    pub segment: &'graph Segment,
    /// If present, the commit that started traversal and the segment that owns it.
    ///
    /// This can differ from [`Self::segment`] when the entrypoint sits on an empty synthetic
    /// segment, such as a virtual workspace tip segment, while the commit lives below it.
    ///
    /// May be `None` if the entrypoint was a reference in a newly born repository,
    /// which doesn't have any commits.
    pub commit_and_owner: Option<(&'graph Commit, &'graph Segment)>,
}

impl<'graph> EntryPoint<'graph> {
    /// The commit that started traversal, if the entrypoint is not unborn.
    pub fn commit(&self) -> Option<&'graph Commit> {
        self.commit_and_owner.map(|(c, _)| c)
    }
}
