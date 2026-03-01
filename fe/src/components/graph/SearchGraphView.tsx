import { useMemo, useState } from 'react';
import type { SearchGraphData, SearchGraphNode, SearchGraphEdge } from '../../types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SearchGraphViewProps {
  searchGraph: SearchGraphData;
}

/** Internal layout node with computed position */
interface LayoutNode {
  node: SearchGraphNode;
  x: number;
  y: number;
  depth: number;
  children: LayoutNode[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NODE_WIDTH = 220;
const NODE_HEIGHT = 80;
const H_GAP = 60;
const V_GAP = 32;

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

function statusIndicator(status: string, evidenceCount: number) {
  // completed with evidence -> green, failed/dead-end -> red, pruned -> gray, running -> blue spinner
  if (status === 'running' || status === 'in_progress') {
    return (
      <span className="relative flex h-3 w-3">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500" />
      </span>
    );
  }
  if (status === 'pruned' || status === 'skipped') {
    return <span className="inline-block h-3 w-3 rounded-full bg-gray-400" />;
  }
  if (status === 'failed' || status === 'dead_end' || status === 'error') {
    return <span className="inline-block h-3 w-3 rounded-full bg-red-500" />;
  }
  // completed
  if (evidenceCount > 0) {
    return <span className="inline-block h-3 w-3 rounded-full bg-green-500" />;
  }
  return <span className="inline-block h-3 w-3 rounded-full bg-gray-300" />;
}

function statusBorder(status: string, evidenceCount: number): string {
  if (status === 'running' || status === 'in_progress') return 'border-blue-400';
  if (status === 'pruned' || status === 'skipped') return 'border-gray-300';
  if (status === 'failed' || status === 'dead_end' || status === 'error') return 'border-red-300';
  if (evidenceCount > 0) return 'border-green-300';
  return 'border-gray-200';
}

// ---------------------------------------------------------------------------
// Tree layout builder
// ---------------------------------------------------------------------------

function buildTree(
  nodes: SearchGraphNode[],
  edges: SearchGraphEdge[],
  rootIds: string[],
): LayoutNode[] {
  const nodeMap = new Map<string, SearchGraphNode>();
  for (const n of nodes) nodeMap.set(n.id, n);

  // Build adjacency from edges
  const childrenMap = new Map<string, string[]>();
  for (const e of edges) {
    const children = childrenMap.get(e.source_id) ?? [];
    children.push(e.target_id);
    childrenMap.set(e.source_id, children);
  }

  // Also find roots via parent_id if root_ids is empty
  const effectiveRoots =
    rootIds.length > 0
      ? rootIds
      : nodes.filter((n) => !n.parent_id).map((n) => n.id);

  // Build layout nodes recursively
  const visited = new Set<string>();

  function buildLayoutNode(id: string, depth: number): LayoutNode | null {
    if (visited.has(id)) return null;
    visited.add(id);
    const node = nodeMap.get(id);
    if (!node) return null;

    const childIds = childrenMap.get(id) ?? [];
    const children: LayoutNode[] = [];
    for (const cid of childIds) {
      const child = buildLayoutNode(cid, depth + 1);
      if (child) children.push(child);
    }

    return { node, x: 0, y: 0, depth, children };
  }

  const roots: LayoutNode[] = [];
  for (const rid of effectiveRoots) {
    const r = buildLayoutNode(rid, 0);
    if (r) roots.push(r);
  }

  // Also add orphan nodes not reachable from roots
  for (const n of nodes) {
    if (!visited.has(n.id)) {
      const orphan = buildLayoutNode(n.id, 0);
      if (orphan) roots.push(orphan);
    }
  }

  return roots;
}

/** Assign x/y positions. Depth determines x, leaf index determines y. */
function assignPositions(roots: LayoutNode[]): { flatNodes: LayoutNode[]; maxX: number; maxY: number } {
  const flatNodes: LayoutNode[] = [];
  let leafIndex = 0;

  function measure(ln: LayoutNode): { minY: number; maxY: number } {
    if (ln.children.length === 0) {
      // Leaf node
      ln.x = ln.depth * (NODE_WIDTH + H_GAP);
      ln.y = leafIndex * (NODE_HEIGHT + V_GAP);
      leafIndex++;
      flatNodes.push(ln);
      return { minY: ln.y, maxY: ln.y };
    }

    let cMinY = Infinity;
    let cMaxY = -Infinity;
    for (const child of ln.children) {
      const { minY, maxY } = measure(child);
      cMinY = Math.min(cMinY, minY);
      cMaxY = Math.max(cMaxY, maxY);
    }

    ln.x = ln.depth * (NODE_WIDTH + H_GAP);
    ln.y = (cMinY + cMaxY) / 2;
    flatNodes.push(ln);
    return { minY: cMinY, maxY: cMaxY };
  }

  for (const root of roots) {
    measure(root);
  }

  let maxX = 0;
  let maxY = 0;
  for (const n of flatNodes) {
    maxX = Math.max(maxX, n.x + NODE_WIDTH);
    maxY = Math.max(maxY, n.y + NODE_HEIGHT);
  }

  return { flatNodes, maxX, maxY };
}

// ---------------------------------------------------------------------------
// Edge lines (SVG)
// ---------------------------------------------------------------------------

function EdgeLines({ flatNodes }: { flatNodes: LayoutNode[] }) {
  const lines: { x1: number; y1: number; x2: number; y2: number; key: string }[] = [];

  for (const ln of flatNodes) {
    for (const child of ln.children) {
      lines.push({
        x1: ln.x + NODE_WIDTH,
        y1: ln.y + NODE_HEIGHT / 2,
        x2: child.x,
        y2: child.y + NODE_HEIGHT / 2,
        key: `${ln.node.id}-${child.node.id}`,
      });
    }
  }

  return (
    <>
      {lines.map(({ x1, y1, x2, y2, key }) => {
        // Bezier curve for nicer look
        const midX = (x1 + x2) / 2;
        const d = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
        return (
          <path
            key={key}
            d={d}
            fill="none"
            stroke="#CBD5E1"
            strokeWidth={2}
            markerEnd="url(#arrowhead)"
          />
        );
      })}
    </>
  );
}

// ---------------------------------------------------------------------------
// Node card
// ---------------------------------------------------------------------------

function NodeCard({
  layoutNode,
  selected,
  onSelect,
}: {
  layoutNode: LayoutNode;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const { node, x, y } = layoutNode;
  const truncatedQuery =
    node.query.length > 50 ? node.query.slice(0, 47) + '...' : node.query;

  return (
    <div
      className={`absolute rounded-lg border-2 bg-white shadow-sm p-3 cursor-pointer transition-shadow hover:shadow-md ${statusBorder(node.status, node.evidence_count)} ${selected ? 'ring-2 ring-blue-500 ring-offset-1' : ''}`}
      style={{
        left: x,
        top: y,
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
      }}
      role="button"
      tabIndex={0}
      onClick={() => onSelect(node.id)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect(node.id);
        }
      }}
      title={node.query}
    >
      {/* Top row: status + query */}
      <div className="flex items-start gap-2 mb-1.5">
        <span className="flex-shrink-0 mt-0.5">{statusIndicator(node.status, node.evidence_count)}</span>
        <span className="text-xs font-medium text-gray-800 leading-tight line-clamp-2">
          {truncatedQuery}
        </span>
      </div>

      {/* Bottom row: badges */}
      <div className="flex items-center gap-2 mt-auto">
        {node.evidence_count > 0 && (
          <span className="inline-flex items-center rounded-full bg-blue-50 px-1.5 py-0.5 text-[10px] font-semibold text-blue-700">
            {node.evidence_count} ev
          </span>
        )}
        {node.best_confidence > 0 && (
          <span
            className={`inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${
              node.best_confidence > 0.7
                ? 'bg-green-50 text-green-700'
                : node.best_confidence > 0.4
                  ? 'bg-yellow-50 text-yellow-700'
                  : 'bg-red-50 text-red-700'
            }`}
          >
            {(node.best_confidence * 100).toFixed(0)}%
          </span>
        )}
        {node.provider && (
          <span className="text-[10px] text-gray-400 font-mono truncate">
            {node.provider}
          </span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Selected node detail panel
// ---------------------------------------------------------------------------

function NodeDetail({ node }: { node: SearchGraphNode }) {
  return (
    <div className="border-t border-gray-200 bg-gray-50 px-4 py-3 space-y-2">
      <h4 className="text-sm font-semibold text-gray-900">Node Detail</h4>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <div>
          <span className="text-gray-500">Query:</span>
          <p className="text-gray-800 break-words">{node.query}</p>
        </div>
        <div>
          <span className="text-gray-500">Intent:</span>
          <p className="text-gray-800">{node.intent || '--'}</p>
        </div>
        <div>
          <span className="text-gray-500">Status:</span>
          <p className="text-gray-800">{node.status}</p>
        </div>
        <div>
          <span className="text-gray-500">Provider:</span>
          <p className="text-gray-800">{node.provider || '--'}</p>
        </div>
        <div>
          <span className="text-gray-500">Evidence:</span>
          <p className="text-gray-800">{node.evidence_count}</p>
        </div>
        <div>
          <span className="text-gray-500">Best Confidence:</span>
          <p className="text-gray-800">
            {node.best_confidence > 0 ? `${(node.best_confidence * 100).toFixed(1)}%` : '--'}
          </p>
        </div>
        <div>
          <span className="text-gray-500">Duration:</span>
          <p className="text-gray-800">
            {node.duration_ms > 0 ? `${(node.duration_ms / 1000).toFixed(2)}s` : '--'}
          </p>
        </div>
        <div>
          <span className="text-gray-500">ID:</span>
          <p className="text-gray-800 font-mono truncate" title={node.id}>
            {node.id.slice(0, 12)}...
          </p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stats summary
// ---------------------------------------------------------------------------

function StatsSummary({ nodes, stats }: { nodes: SearchGraphNode[]; stats: Record<string, unknown> }) {
  const total = nodes.length;
  const completed = nodes.filter(
    (n) => n.status === 'completed' || n.status === 'done',
  ).length;
  const failed = nodes.filter(
    (n) => n.status === 'failed' || n.status === 'error' || n.status === 'dead_end',
  ).length;
  const totalEvidence = nodes.reduce((sum, n) => sum + n.evidence_count, 0);

  const items = [
    { label: 'Total Nodes', value: total, color: 'text-gray-900' },
    { label: 'Completed', value: completed, color: 'text-green-600' },
    { label: 'Failed', value: failed, color: 'text-red-600' },
    { label: 'Total Evidence', value: totalEvidence, color: 'text-blue-600' },
  ];

  // Merge any extra stats from the backend
  for (const [key, val] of Object.entries(stats)) {
    if (typeof val === 'number') {
      items.push({
        label: key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
        value: val,
        color: 'text-gray-700',
      });
    }
  }

  return (
    <div className="flex flex-wrap gap-4 px-4 py-3 bg-gray-50 border-t border-gray-200">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-1.5 text-xs">
          <span className="text-gray-500">{item.label}:</span>
          <span className={`font-semibold ${item.color}`}>{item.value}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SearchGraphView
// ---------------------------------------------------------------------------

export default function SearchGraphView({ searchGraph }: SearchGraphViewProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const { flatNodes, maxX, maxY } = useMemo(() => {
    const trees = buildTree(searchGraph.nodes, searchGraph.edges, searchGraph.root_ids);
    return assignPositions(trees);
  }, [searchGraph]);

  const selectedNode = useMemo(
    () => searchGraph.nodes.find((n) => n.id === selectedNodeId) ?? null,
    [searchGraph.nodes, selectedNodeId],
  );

  if (searchGraph.nodes.length === 0) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
        <p className="text-sm text-gray-500">No search graph data available yet.</p>
      </div>
    );
  }

  const canvasWidth = maxX + 40;
  const canvasHeight = maxY + 40;

  return (
    <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <h3 className="text-sm font-semibold text-gray-900">Search Graph</h3>
        <p className="text-xs text-gray-500 mt-0.5">
          Click a node to view details
        </p>
      </div>

      {/* Graph canvas */}
      <div
        className="overflow-auto relative"
        style={{ maxHeight: 480 }}
      >
        <div
          className="relative"
          style={{
            width: canvasWidth,
            height: canvasHeight,
            minWidth: '100%',
            padding: 20,
          }}
        >
          {/* SVG edges */}
          <svg
            className="absolute inset-0 pointer-events-none"
            width={canvasWidth}
            height={canvasHeight}
            style={{ left: 0, top: 0 }}
          >
            <defs>
              <marker
                id="arrowhead"
                markerWidth="8"
                markerHeight="6"
                refX="8"
                refY="3"
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill="#CBD5E1" />
              </marker>
            </defs>
            <EdgeLines flatNodes={flatNodes} />
          </svg>

          {/* Node cards */}
          {flatNodes.map((ln) => (
            <NodeCard
              key={ln.node.id}
              layoutNode={ln}
              selected={ln.node.id === selectedNodeId}
              onSelect={setSelectedNodeId}
            />
          ))}
        </div>
      </div>

      {/* Detail panel */}
      {selectedNode && <NodeDetail node={selectedNode} />}

      {/* Stats */}
      <StatsSummary nodes={searchGraph.nodes} stats={searchGraph.stats as Record<string, unknown>} />
    </div>
  );
}
