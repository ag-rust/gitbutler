//! The segment arena and its connections, where each [`Segment`] directly owns its outgoing
//! [`Connection`]s.
//!
//! There is no separate edge list and no generic graph container: the segments *are* the graph.
//! [`Graph`] holds an arena of segments (`Vec<Option<Segment>>` with a free-list so ids stay
//! stable across removal), and every connection — which target segment, and which commit
//! connects to which — lives on the source segment in [`Segment::connections`].
//!
//! Outgoing connections are kept in first-parent order (see
//! `order_outgoing_connections_by_first_parent`), which is the order traversal wants. The reverse
//! direction is a maintained index (`incoming`: one source entry per connection), so per-segment
//! incoming queries are O(degree) even on graphs with hundreds of thousands of segments.

use crate::{Graph, Segment};

/// Which side of a segment's connections to traverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Connections leaving the segment (stored on it).
    Outgoing,
    /// Connections entering the segment (derived by scanning).
    Incoming,
}

/// An outgoing connection from one [`Segment`] to another, stored on the source segment.
///
/// It carries the intent of the connection: *which* commit in the source connects to *which* commit
/// in the target, by id. In a finished, validated graph an endpoint id is present exactly when the
/// respective segment has commits, and then it is the source's *last* commit and the target's
/// *first* — so the commit indices are derived, not stored.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Connection {
    /// The segment this connection points to.
    pub target: usize,
    /// The last commit of the source segment, `None` if the source has no commit.
    pub(crate) src_id: Option<gix::ObjectId>,
    /// The first commit of the target segment, `None` if the target has no commit.
    pub(crate) dst_id: Option<gix::ObjectId>,
}

impl Connection {
    /// Create a connection to `target` carrying the given commit endpoints.
    pub(crate) fn new(
        target: usize,
        src_id: Option<gix::ObjectId>,
        dst_id: Option<gix::ObjectId>,
    ) -> Self {
        Connection {
            target,
            src_id,
            dst_id,
        }
    }

    /// The parent commit this connection points at; `None` for synthetic connections with no concrete commit.
    pub fn dst_id(&self) -> Option<gix::ObjectId> {
        self.dst_id
    }

    /// The source commit id this connection emanates from, if it starts at a concrete commit.
    pub fn src_id(&self) -> Option<gix::ObjectId> {
        self.src_id
    }
}

/// A connection together with the segment it leaves, yielded while iterating a segment's edges.
#[derive(Debug, Clone, Copy)]
pub struct EdgeRef<'a> {
    /// The segment this connection leaves.
    pub source: usize,
    /// The segment this connection points to.
    pub target: usize,
    /// The connection payload.
    pub connection: &'a Connection,
}

/// The segment-arena methods: id-stable insertion/removal and connection queries.
impl Graph {
    /// Number of live (non-tombstoned) segments.
    pub(crate) fn segment_count(&self) -> usize {
        self.segments.iter().filter(|n| n.is_some()).count()
    }

    /// Insert `segment` and return its id, reusing a tombstoned slot if available.
    /// The caller is responsible for having set `segment.id` to the returned id where it matters.
    pub fn add_segment(&mut self, segment: Segment) -> usize {
        if let Some(id) = self.free.pop() {
            debug_assert!(self.incoming[id].is_empty(), "tombstones have no edges");
            self.segments[id] = Some(segment);
            id
        } else {
            let id = self.segments.len();
            self.segments.push(Some(segment));
            self.incoming.push(Vec::new());
            id
        }
    }

    /// The segment at `id`, if live.
    pub fn segment(&self, id: usize) -> Option<&Segment> {
        self.segments.get(id).and_then(Option::as_ref)
    }

    /// The segment at `id` mutably, if live.
    pub(crate) fn segment_mut(&mut self, id: usize) -> Option<&mut Segment> {
        self.segments.get_mut(id).and_then(Option::as_mut)
    }

    /// Tombstone `id` and drop every connection incident to it (its own, and those pointing at it).
    pub(crate) fn remove_segment(&mut self, id: usize) -> Option<Segment> {
        let removed = self.segments.get_mut(id)?.take()?;
        for c in &removed.connections {
            let inc = &mut self.incoming[c.target];
            let pos = inc.iter().position(|&s| s == id).expect("index in sync");
            inc.swap_remove(pos);
        }
        for src in std::mem::take(&mut self.incoming[id]) {
            if let Some(seg) = self.segments.get_mut(src).and_then(Option::as_mut) {
                seg.connections.retain(|c| c.target != id);
            }
        }
        self.free.push(id);
        Some(removed)
    }

    /// Add a connection `source -> target`, appended to `source`'s connections.
    pub fn add_edge(&mut self, source: usize, connection: Connection) {
        debug_assert!(
            self.segment(source).is_some(),
            "connection source must be live"
        );
        debug_assert!(
            self.segment(connection.target).is_some(),
            "connection target must be live"
        );
        self.incoming[connection.target].push(source);
        self.segments[source]
            .as_mut()
            .expect("live source")
            .connections
            .push(connection);
    }

    /// Remove the first connection leaving `source` that equals `connection`, returning it.
    /// Connections have no global id; they are identified by source plus value.
    pub fn remove_edge(&mut self, source: usize, connection: &Connection) -> Option<Connection> {
        let conns = &mut self.segment_mut(source)?.connections;
        let pos = conns.iter().position(|c| c == connection)?;
        let removed = conns.remove(pos);
        let inc = &mut self.incoming[removed.target];
        let pos = inc
            .iter()
            .position(|&s| s == source)
            .expect("index in sync");
        inc.swap_remove(pos);
        Some(removed)
    }

    /// Live segment ids, ascending.
    pub fn segment_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.segments
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.as_ref().map(|_| i))
    }

    /// Connections leaving (`Outgoing`, in stored first-parent order) or entering (`Incoming`,
    /// via the incoming index, in ascending source order) `segment_id`.
    pub fn edges_directed(
        &self,
        segment_id: usize,
        dir: Direction,
    ) -> impl Iterator<Item = EdgeRef<'_>> + '_ {
        // Collect so the two directions share a return type without boxing.
        let mut out = Vec::new();
        match dir {
            Direction::Outgoing => {
                if let Some(seg) = self.segment(segment_id) {
                    for c in &seg.connections {
                        out.push(EdgeRef {
                            source: segment_id,
                            target: c.target,
                            connection: c,
                        });
                    }
                }
            }
            Direction::Incoming => {
                let mut sources = self.incoming.get(segment_id).cloned().unwrap_or_default();
                sources.sort_unstable();
                sources.dedup();
                for src in sources {
                    for c in &self.segments[src].as_ref().expect("live").connections {
                        if c.target == segment_id {
                            out.push(EdgeRef {
                                source: src,
                                target: c.target,
                                connection: c,
                            });
                        }
                    }
                }
            }
        }
        out.into_iter()
    }

    /// Neighboring segment ids in the given direction.
    pub(crate) fn neighbors_directed(
        &self,
        segment_id: usize,
        dir: Direction,
    ) -> impl Iterator<Item = usize> + '_ {
        self.edges_directed(segment_id, dir)
            .map(move |e| match dir {
                Direction::Outgoing => e.target,
                Direction::Incoming => e.source,
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Topological order (Kahn's algorithm): every connection's source precedes its target.
    /// Assumes a DAG; segments left over after a cycle are omitted.
    pub(crate) fn toposort(&self) -> Vec<usize> {
        let mut in_degree: Vec<usize> = self.incoming.iter().map(Vec::len).collect();
        let mut queue: std::collections::VecDeque<usize> = self
            .segment_ids()
            .filter(|&id| in_degree[id] == 0)
            .collect();
        let mut order = Vec::with_capacity(self.segment_count());
        while let Some(id) = queue.pop_front() {
            order.push(id);
            for c in &self.segments[id].as_ref().expect("live").connections {
                in_degree[c.target] -= 1;
                if in_degree[c.target] == 0 {
                    queue.push_back(c.target);
                }
            }
        }
        order
    }

    /// Live segments with no connections in `dir` — sinks for `Outgoing`, roots for `Incoming`.
    pub fn externals(&self, dir: Direction) -> impl Iterator<Item = usize> + '_ {
        self.segment_ids()
            .filter(move |&id| match dir {
                Direction::Outgoing => self.segments[id]
                    .as_ref()
                    .expect("live")
                    .connections
                    .is_empty(),
                Direction::Incoming => self.incoming[id].is_empty(),
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Number of live connections.
    pub(crate) fn edge_count(&self) -> usize {
        self.segments
            .iter()
            .flatten()
            .map(|s| s.connections.len())
            .sum()
    }

    /// All live connections, in ascending source order.
    pub fn edge_references(&self) -> impl Iterator<Item = EdgeRef<'_>> + '_ {
        let mut out = Vec::new();
        for src in self.segment_ids() {
            for c in &self.segments[src].as_ref().expect("live").connections {
                out.push(EdgeRef {
                    source: src,
                    target: c.target,
                    connection: c,
                });
            }
        }
        out.into_iter()
    }

    /// One past the largest segment id ever handed out (tombstoned slots included).
    pub(crate) fn id_bound(&self) -> usize {
        self.segments.len()
    }

    /// All live segments, ascending by id.
    pub(crate) fn segments(&self) -> impl Iterator<Item = &Segment> + '_ {
        self.segments.iter().filter_map(Option::as_ref)
    }

    /// All live segments mutably, ascending by id.
    pub(crate) fn segments_mut(&mut self) -> impl Iterator<Item = &mut Segment> + '_ {
        self.segments.iter_mut().filter_map(Option::as_mut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Segment;

    fn conn(target: usize) -> Connection {
        Connection::new(target, None, None)
    }

    #[test]
    fn add_remove_and_lifo_id_reuse() {
        let mut g = Graph::default();
        let a = g.add_segment(Segment::default());
        let b = g.add_segment(Segment::default());
        assert_eq!(g.segment_count(), 2);
        assert!(g.remove_segment(a).is_some());
        assert_eq!(g.segment_count(), 1);
        // The freed slot is reused before a fresh id is handed out.
        let c = g.add_segment(Segment::default());
        assert_eq!(c, a, "freed slot reused");
        assert!(g.segment(b).is_some(), "other ids stay stable");
    }

    #[test]
    fn outgoing_is_stored_order_and_incoming_is_scanned() {
        let mut g = Graph::default();
        let a = g.add_segment(Segment::default());
        let b = g.add_segment(Segment::default());
        let c = g.add_segment(Segment::default());
        g.add_edge(a, conn(b));
        g.add_edge(a, conn(c));
        // Outgoing keeps insertion (stored) order — no newest-first reversal.
        assert_eq!(
            g.neighbors_directed(a, Direction::Outgoing)
                .collect::<Vec<_>>(),
            vec![b, c]
        );
        assert_eq!(
            g.neighbors_directed(c, Direction::Incoming)
                .collect::<Vec<_>>(),
            vec![a]
        );
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn remove_segment_drops_incident_connections() {
        let mut g = Graph::default();
        let a = g.add_segment(Segment::default());
        let b = g.add_segment(Segment::default());
        let c = g.add_segment(Segment::default());
        g.add_edge(a, conn(b));
        g.add_edge(b, conn(c));
        g.remove_segment(b);
        assert_eq!(g.neighbors_directed(a, Direction::Outgoing).count(), 0);
        assert_eq!(g.neighbors_directed(c, Direction::Incoming).count(), 0);
        assert!(g.segment(b).is_none());
    }

    #[test]
    fn remove_edge_by_value() {
        let mut g = Graph::default();
        let a = g.add_segment(Segment::default());
        let b = g.add_segment(Segment::default());
        let e = conn(b);
        g.add_edge(a, e);
        assert_eq!(g.edge_count(), 1);
        assert!(g.remove_edge(a, &e).is_some());
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn toposort_orders_sources_before_targets() {
        // Diamond: a -> b, a -> c, b -> d, c -> d.
        let mut g = Graph::default();
        let a = g.add_segment(Segment::default());
        let b = g.add_segment(Segment::default());
        let c = g.add_segment(Segment::default());
        let d = g.add_segment(Segment::default());
        g.add_edge(a, conn(b));
        g.add_edge(a, conn(c));
        g.add_edge(b, conn(d));
        g.add_edge(c, conn(d));
        let order = g.toposort();
        assert_eq!(order.len(), 4);
        let pos = |n: usize| order.iter().position(|&x| x == n).unwrap();
        assert!(pos(a) < pos(b) && pos(a) < pos(c) && pos(b) < pos(d) && pos(c) < pos(d));
    }

    #[test]
    fn externals_finds_roots_and_sinks() {
        let mut g = Graph::default();
        let a = g.add_segment(Segment::default());
        let b = g.add_segment(Segment::default());
        let c = g.add_segment(Segment::default());
        g.add_edge(a, conn(b));
        g.add_edge(b, conn(c));
        assert_eq!(
            g.externals(Direction::Incoming).collect::<Vec<_>>(),
            vec![a]
        );
        assert_eq!(
            g.externals(Direction::Outgoing).collect::<Vec<_>>(),
            vec![c]
        );
    }
}
