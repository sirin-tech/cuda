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
  Returns node in the graph by its name
  """
  @spec node(graph :: Graph.t, id :: Graph.id) :: Node.t
  def node(graph, id)
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

  def node(%{nodes: nodes}, id) do
    nodes |> Enum.find(fn
      %{id: ^id} -> true
      _          -> false
    end)
  end
  def node(_, _), do: nil
end
