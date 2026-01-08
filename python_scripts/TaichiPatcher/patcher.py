import numpy as np
import os
import tempfile
from typing import Tuple, Optional

# Optional imports for different approaches
try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False
    print("Warning: meshio not installed. Install with: pip install meshio")

try:
    import pymetis
    HAS_PYMETIS = True
except ImportError:
    HAS_PYMETIS = False

try:
    import taichi as ti
    import meshtaichi_patcher as Patcher
    HAS_MESHTAICHI = True
except ImportError:
    HAS_MESHTAICHI = False


def load_mesh_with_meshio(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a tetrahedral mesh using meshio (more robust parser).
    
    Returns:
        vertices: (N, 3) array of vertex positions
        cells: (M, 4) array of tetrahedral cell indices
    """
    if not HAS_MESHIO:
        raise ImportError("meshio is required. Install with: pip install meshio")
    
    mesh = meshio.read(mesh_path)
    vertices = mesh.points
    
    # Find tetrahedral cells
    cells = None
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            cells = cell_block.data
            break
    
    if cells is None:
        raise ValueError(f"No tetrahedral cells found in {mesh_path}")
    
    print(f"Loaded mesh with meshio:")
    print(f"  Vertices: {vertices.shape[0]}")
    print(f"  Tetrahedra: {cells.shape[0]}")
    
    return vertices, cells


def convert_mesh_to_vtk(mesh_path: str, output_path: Optional[str] = None) -> str:
    """Convert a mesh file to VTK format using meshio.
    
    Args:
        mesh_path: Path to the input mesh file
        output_path: Optional path for the output VTK file
        
    Returns:
        Path to the VTK file
    """
    if not HAS_MESHIO:
        raise ImportError("meshio is required. Install with: pip install meshio")
    
    mesh = meshio.read(mesh_path)
    
    if output_path is None:
        # Create temp file
        fd, output_path = tempfile.mkstemp(suffix=".vtk")
        os.close(fd)
    
    meshio.write(output_path, mesh, file_format="vtk")
    print(f"Converted mesh to VTK: {output_path}")
    
    return output_path


def build_vertex_adjacency(cells: np.ndarray, n_verts: int) -> list:
    """Build vertex-to-vertex adjacency list from tetrahedral cells."""
    from collections import defaultdict
    adj = defaultdict(set)
    
    # For each tetrahedron, connect all pairs of vertices
    for tet in cells:
        for i in range(4):
            for j in range(i + 1, 4):
                adj[tet[i]].add(tet[j])
                adj[tet[j]].add(tet[i])
    
    # Convert to list format for METIS
    adjacency = [list(adj[i]) for i in range(n_verts)]
    return adjacency


def partition_with_metis(adjacency: list, n_parts: int) -> np.ndarray:
    """Partition vertices using METIS graph partitioning."""
    if not HAS_PYMETIS:
        raise ImportError("pymetis is required. Install with: pip install pymetis")
    
    # Use Pythonic adjacency list (avoids deprecation warning for xadj/adjncy)
    n_cuts, membership = pymetis.part_graph(n_parts, adjacency=adjacency)
    
    print(f"METIS partitioning: {n_parts} parts, {n_cuts} edge cuts")
    return np.array(membership, dtype=np.int32)


def partition_with_scipy(adjacency: list, n_parts: int) -> np.ndarray:
    """Partition vertices using scipy's spectral clustering (fallback)."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.cluster import KMeans
    
    n_verts = len(adjacency)
    
    # Build sparse adjacency matrix
    adj_matrix = lil_matrix((n_verts, n_verts), dtype=np.float32)
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            adj_matrix[i, j] = 1.0
    adj_matrix = adj_matrix.tocsr()
    
    # Use simple spatial partitioning as fallback
    # This is less optimal than METIS but works without extra dependencies
    print(f"Using scipy KMeans clustering as fallback (less optimal than METIS)")
    
    # Get vertex coordinates if available, otherwise use random projection
    from scipy.sparse.linalg import eigsh
    try:
        # Compute Laplacian eigenvectors for spectral clustering
        degree = np.array(adj_matrix.sum(axis=1)).flatten()
        D_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-10))
        L_norm = lil_matrix((n_verts, n_verts), dtype=np.float32)
        L_norm.setdiag(1.0)
        for i, neighbors in enumerate(adjacency):
            for j in neighbors:
                L_norm[i, j] = -D_inv_sqrt[i] * D_inv_sqrt[j]
        L_norm = L_norm.tocsr()
        
        # Get first k eigenvectors
        k = min(n_parts, 10)
        eigenvalues, eigenvectors = eigsh(L_norm, k=k, which='SM')
        
        # Cluster using KMeans
        kmeans = KMeans(n_clusters=n_parts, random_state=42, n_init=10)
        membership = kmeans.fit_predict(eigenvectors)
    except Exception as e:
        print(f"Spectral clustering failed: {e}, using simple round-robin")
        membership = np.arange(n_verts) % n_parts
    
    return np.array(membership, dtype=np.int32)


def compute_vertex_patch_ids_fallback(
    mesh_path: str, 
    n_parts: Optional[int] = None,
    patch_size: Optional[int] = None
) -> np.ndarray:
    """Compute vertex patch IDs using meshio + METIS/scipy fallback.
    
    Args:
        mesh_path: Path to the mesh file
        n_parts: Number of partitions (takes priority over patch_size if specified)
        patch_size: Target vertices per patch (used to auto-calculate n_parts)
                   Default is 1024 if neither n_parts nor patch_size is specified
    
    Returns:
        patch_ids: Array of patch IDs for each vertex
    """
    # Load mesh with meshio
    vertices, cells = load_mesh_with_meshio(mesh_path)
    n_verts = vertices.shape[0]
    
    # Calculate number of partitions
    if n_parts is not None:
        # Use explicit n_parts
        pass
    elif patch_size is not None:
        # Calculate from patch_size
        n_parts = max(1, n_verts // patch_size)
    else:
        # Default: 1024 vertices per patch
        n_parts = max(1, n_verts // 1024)
    
    print(f"Partitioning {n_verts} vertices into {n_parts} parts (~{n_verts // n_parts} verts/patch)")
    
    # Build adjacency
    print("Building vertex adjacency graph...")
    adjacency = build_vertex_adjacency(cells, n_verts)
    
    # Partition
    if HAS_PYMETIS:
        patch_ids = partition_with_metis(adjacency, n_parts)
    else:
        print("pymetis not available, using scipy fallback")
        patch_ids = partition_with_scipy(adjacency, n_parts)
    
    return patch_ids


def load_tet_mesh_meshtaichi(mesh_path: str, convert_via_vtk: bool = True):
    """Load mesh using meshtaichi_patcher with optional VTK conversion.
    
    If direct loading fails (common with .mesh files), converts via VTK format.
    
    Args:
        mesh_path: Path to the mesh file
        convert_via_vtk: If True, convert to VTK first (more reliable for .mesh files)
    """
    if not HAS_MESHTAICHI:
        raise ImportError("meshtaichi_patcher not available")
    
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    # Initialize Taichi
    ti.init(arch=ti.cuda)
    
    load_path = mesh_path
    temp_vtk = None
    
    # Convert to VTK if requested (helps with .mesh files that meshtaichi can't parse)
    if convert_via_vtk and HAS_MESHIO:
        print("Converting mesh to VTK format for better compatibility...")
        temp_vtk = convert_mesh_to_vtk(mesh_path)
        load_path = temp_vtk
    
    try:
        mesh = Patcher.load_mesh(
            load_path, 
            relations=["CV", "VC"],
            max_order=3
        )
        
        print(f"Loaded mesh with meshtaichi:")
        print(f"  Vertices: {len(mesh.verts)}")
        print(f"  Cells: {len(mesh.cells)}")
        
        mesh.verts.place({"pos": ti.math.vec3})
        mesh.verts.pos.from_numpy(mesh.get_position_as_numpy())
        
        return mesh
    finally:
        # Clean up temp file
        if temp_vtk and os.path.exists(temp_vtk):
            os.remove(temp_vtk)


def compute_vertex_patch_ids_meshtaichi(mesh) -> np.ndarray:
    """Compute patch IDs using meshtaichi (original approach)."""
    pos_np = mesh.get_position_as_numpy()
    n_verts = pos_np.shape[0]

    patch_id = ti.field(dtype=ti.i32, shape=n_verts)

    @ti.kernel
    def _kernel():
        for v in mesh.verts:
            patch_id[v.id] = ti.mesh_patch_idx()

    _kernel()
    return patch_id.to_numpy()


def compute_vertex_patch_ids(
    mesh_path: str, 
    n_parts: Optional[int] = None,
    patch_size: Optional[int] = None,
    use_taichi: bool = True,
    fallback_to_metis: bool = True
) -> np.ndarray:
    """Compute vertex patch IDs for a tetrahedral mesh.
    
    Uses Taichi's native mesh partitioning (via meshtaichi_patcher).
    Falls back to METIS if Taichi fails and fallback_to_metis=True.
    
    Args:
        mesh_path: Path to the mesh file (.mesh, .vtk, etc.)
        n_parts: Number of partitions (takes priority over patch_size if specified)
                 Only used for METIS fallback; Taichi determines partitions automatically
        patch_size: Target vertices per patch (used to auto-calculate n_parts for METIS)
                   Default is 1024 if neither n_parts nor patch_size is specified
        use_taichi: Whether to use Taichi partitioning (default: True)
        fallback_to_metis: Whether to fall back to METIS if Taichi fails (default: True)
    
    Returns:
        patch_ids: Array of patch IDs for each vertex
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    # Try Taichi partitioning first
    if use_taichi and HAS_MESHTAICHI:
        try:
            print("Using Taichi mesh partitioning...")
            # Convert via VTK for .mesh files (meshtaichi can't parse them directly)
            convert_via_vtk = mesh_path.lower().endswith('.mesh')
            mesh = load_tet_mesh_meshtaichi(mesh_path, convert_via_vtk=convert_via_vtk)
            return compute_vertex_patch_ids_meshtaichi(mesh)
        except Exception as e:
            print(f"Taichi partitioning failed: {e}")
            if fallback_to_metis:
                print("Falling back to METIS approach...")
            else:
                raise
    
    # Fallback to meshio + METIS
    if fallback_to_metis:
        return compute_vertex_patch_ids_fallback(mesh_path, n_parts, patch_size)
    else:
        raise RuntimeError("Taichi partitioning failed and fallback_to_metis=False")


# --- example ---
if __name__ == "__main__":
    tet_mesh_path = "/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tetMeshes/498457_tetmesh.mesh"
    
    # Compute patch IDs using Taichi's native mesh partitioning
    # The .mesh file is automatically converted to VTK for better compatibility
    # 
    # Options:
    #   use_taichi=True (default): Use Taichi's native partitioning
    #   use_taichi=False:          Use METIS partitioning directly
    #   fallback_to_metis=True:    Fall back to METIS if Taichi fails (default)
    #   fallback_to_metis=False:   Raise error if Taichi fails
    #
    # For METIS fallback only:
    #   patch_size=1024: ~1024 vertices per patch (default)
    #   n_parts=100:     exactly 100 partitions
    
    patch_ids = compute_vertex_patch_ids(
        tet_mesh_path, 
        use_taichi=True,           # Use Taichi partitioning
        fallback_to_metis=True     # Fall back to METIS if Taichi fails
    )
    
    # Save results
    output_file = "patch_ids_tet_vertex_order.txt"
    np.savetxt(output_file, patch_ids, fmt="%d")
    print(f"Saved: {output_file}, shape: {patch_ids.shape}")
    print(f"Number of unique patches: {len(np.unique(patch_ids))}")