alias Cuda.Graph
alias Cuda.Graph.Node
alias Cuda.Graph.Pin

defprotocol Cuda.Graph.NodeProto do
  @doc """
  Returns pin by its id
  """
  @spec pin(node:: Node.t, id: Graph.id) :: Pin.t | nil
  def pin(node, id)

  @doc """
  Returns a list of pins of specified type
  """
  @spec pins(node :: Node.t, type :: Pin.type | [Pin.type]) :: [Pin.t]
  def pins(node, type \\ nil)

  @spec assign(node :: struct, key :: atom, value :: any) :: struct
  def assign(node, key, value)

  @spec assign(node :: struct, key :: map | keyword) :: struct
  def assign(node, assigns)
end

defprotocol Cuda.Graph.GraphProto do
  @spec add(graph :: Graph.t, node :: Node.t) :: Graph.t
  def add(graph, node)

  @doc """
  Replaces node in the graph.

  If the node to replace have same id as a replaced node, you can call this
  function with two arguments - graph and the node to replace. If you need to
  replace node which id is different from replacing node id, pass id of node
  to replace as second argument and replacement node as a third argument.
  """
  @spec replace(graph :: Graph.t, node :: Node.t) :: Graph.t
  @spec replace(graph :: Graph.t, id :: Graph.id | [Graph.id], node :: Node.t) :: Graph.t
  def replace(graph, node)
  def replace(graph, id, node)

  @doc """
  Returns node in the graph by its name or path (a list of names)
  """
  @spec node(graph :: Graph.t, id :: Graph.id | [Graph.id]) :: Node.t
  def node(graph, id)

  @doc """
  Returns pin of link specification. It can be a pin of graph itself or a pin
  of child node
  """
  @spec link_spec_pin(graph :: Graph.t, link_spec :: Graph.link_spec) :: Pin.t
  def link_spec_pin(graph, link_spec)

  @doc """
  Returns a node of link specification. It can be a graph itself or child node
  """
  @spec link_spec_node(graph :: Graph.t, link_spec :: Graph.link_spec) :: Node.t | Graph.t
  def link_spec_node(graph, link_spec)
end

defprotocol Cuda.Graph.Factory do
  @doc """
  Creates a new evaluation node
  """
  @spec new(node :: struct, id :: Graph.id, module :: atom, opts :: keyword, env :: Cuda.Env.t) :: struct
  def new(node, id, module, opts \\ [], env \\ [])
end

defimpl Cuda.Graph.NodeProto, for: Any do
  def pin(%{pins: pins}, id) do
    pins |> Enum.find(fn
      %Pin{id: ^id} -> true
      _             -> false
    end)
  end
  def get_pin(_, _), do: nil

  def pins(%{pins: pins}, nil), do: pins
  def pins(node, types) when is_list(types) do
    Enum.reduce(types, [], &(&2 ++ pins(node, &1)))
  end
  def pins(%{pins: pins}, type) do
    pins |> Enum.filter(fn
      %Pin{type: ^type} -> true
      _                 -> false
    end)
  end

  def assign(%{assigns: assigns} = node, key, value) do
    %{node | assigns: Map.put(assigns, key, value)}
  end

  def assign(%{assigns: assigns} = node, data) do
    data = data |> Enum.into(%{})
    %{node | assigns: Map.merge(assigns, data)}
  end
end

defimpl Cuda.Graph.GraphProto, for: Any do
  require Cuda
  import Cuda, only: [compile_error: 1]

  def add(%{nodes: nodes} = graph, %{id: id} = node) do
    with nil <- node(graph, id) do
      %{graph | nodes: [node | nodes]}
    else
      _ -> compile_error("Node with id `#{id}` is already in the graph")
    end
  end

  def replace(%{nodes: nodes} = graph, %{id: id} = node) do
    nodes = nodes |> Enum.map(fn
      %{id: ^id} -> node
      x          -> x
    end)
    %{graph | nodes: nodes}
  end
  def replace(%{nodes: _} = graph, [], node), do: replace(graph, node)
  def replace(%{id: src}, [], %{id: dst} = node) when src == dst, do: node
  def replace(graph, [id | path], node) do
    with %{} = child <- Cuda.Graph.GraphProto.node(graph, id) do
      replace(graph, replace(child, path, node))
    end
  end
  def replace(%{nodes: nodes} = graph, id, node) do
    nodes = nodes |> Enum.map(fn
      %{id: ^id} -> node
      x          -> x
    end)
    %{graph | nodes: nodes}
  end

  def node(_, []), do: nil
  def node(%{nodes: _} = graph, [id]), do: node(graph, id)
  def node(%{nodes: _} = graph, [id | path]) do
    with %{} = child <- Cuda.Graph.GraphProto.node(graph, id) do
      Cuda.Graph.GraphProto.node(child, path)
    end
  end
  def node(%{nodes: nodes}, id) do
    nodes |> Enum.find(fn
      %{id: ^id} -> true
      _          -> false
    end)
  end
  def node(_, _), do: nil

  def link_spec_pin(graph, {:__self__, pin}) do
    Cuda.Graph.NodeProto.pin(graph, pin)
  end
  def link_spec_pin(graph, {node, pin}) do
    with %{} = node <- node(graph, node) do
      Cuda.Graph.NodeProto.pin(node, pin)
    end
  end

  def link_spec_node(graph, {:__self__, _}) do
    graph
  end
  def link_spec_node(graph, {node, _}) do
    node(graph, node)
  end
end
