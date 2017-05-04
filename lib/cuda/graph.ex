defmodule Cuda.Graph do
  @moduledoc """
  Represents evaluation graph
  """
  require Cuda
  import Cuda, only: [compile_error: 1]

  alias Cuda.Graph.Pin
  alias Cuda.Graph.Node
  alias Cuda.Graph.GraphProto
  alias Cuda.Graph.NodeProto

  @type id :: String.t | atom | non_neg_integer
  @type link :: {id, id}

  @type t :: %__MODULE__{
    id: id,
    module: module,
    type: Node.type,
    pins: [Pin.t],
    nodes: [Node.t],
    links: [{link, link}],
  }

  @type dfs_action :: :enter | :move | :leave
  @type dfs_result :: {:ok | :error | atom, state :: any}
  @type dfs_callback :: (action :: dfs_action, arg :: any, state :: any -> dfs_result)

  @callback __graph__(graph :: t, opts :: keyword, env :: keyword) :: t

  @derive [NodeProto, GraphProto]
  defstruct [:id, :module, type: :graph, pins: [], nodes: [], links: []]

  @self :__self__
  @input_pins  ~w(input consumer)a
  @output_pins ~w(output producer)a

  @exports [add: 3, add: 4, add: 5, link: 3]

  defmacro __using__(_opts) do
    quote do
      use Cuda.Graph.Node
      import unquote(__MODULE__), only: unquote(@exports)
      @behaviour unquote(__MODULE__)
      def __type__(_, _), do: :graph
      def __proto__(_, _), do: unquote(__MODULE__)
      defoverridable __type__: 2
    end
  end

  def add(%__MODULE__{} = graph, id, module, opts \\ [], env \\ []) do
    with {:module, module} <- Code.ensure_loaded(module) do
      proto = struct(Node.proto(module, opts, env))
      GraphProto.add(graph, Cuda.Graph.Factory.new(proto, id, module, opts, env))
    else
      _ -> compile_error("Graph module #{module} could not be loaded")
    end
  end

  def link(%__MODULE__{links: links} = graph, {sn, sp} = src, {dn, dp} = dst) do
    # node to node connection
    with {:src, %{} = src_node} <- {:src, GraphProto.node(graph, sn)},
         {:dst, %{} = dst_node} <- {:dst, GraphProto.node(graph, dn)} do
      src_pin = assert_pin_type(src_node, sp, @output_pins)
      dst_pin = assert_pin_type(dst_node, dp, @input_pins)
      assert_pin_data_type(src_pin, dst_pin)
      %{graph | links: [{src, dst} | links]}
    else
      {:src, _} -> compile_error("Source node `#{sn}` not found")
      {:dst, _} -> compile_error("Destination node `#{dn}` not found")
    end
  end

  def link(%__MODULE__{links: links} = graph, src, {dn, dp} = dst) do
    # input to node connection
    with %{} = dst_node <- GraphProto.node(graph, dn) do
      src_pin = assert_pin_type(graph, src, @input_pins)
      dst_pin = assert_pin_type(dst_node, dp, @input_pins)
      assert_pin_data_type(src_pin, dst_pin)
      %{graph | links: [{{@self, src}, dst} | links]}
    else
      _ -> compile_error("Destination node `#{dn}` not found")
    end
  end

  def link(%__MODULE__{links: links} = graph, {sn, sp} = src, dst) do
    # node to output connection
    with %{} = src_node <- GraphProto.node(graph, sn) do
      src_pin = assert_pin_type(graph, dst, @output_pins)
      dst_pin = assert_pin_type(src_node, sp, @output_pins)
      assert_pin_data_type(src_pin, dst_pin)
      %{graph | links: [{src, {@self, dst}} | links]}
    else
      _ -> compile_error("Source node `#{sn}` not found")
    end
  end

  def link(%__MODULE__{links: links} = graph, src, dst) do
    # input to output connection
    src_pin = assert_pin_type(graph, src, @input_pins)
    dst_pin = assert_pin_type(graph, dst, @output_pins)
    assert_pin_data_type(src_pin, dst_pin)
    %{graph | links: [{{@self, src}, {@self, dst}} | links]}
  end

  defp assert_pin_type(node, pin_name, types) do
    with %Pin{} = pin <- NodeProto.pin(node, pin_name) do
      if not pin.type in types do
        types = types |> Enum.map(& "#{&1}") |> Enum.join(" or ")
        compile_error("Pin `#{pin_name}` of node `#{node.id}` has a wrong" <>
                      " type. The #{types} types are expected.")
      end
      pin
    else
      _ -> compile_error("Pin `#{pin_name}` not found in node `#{node.id}`")
    end
  end

  defp assert_pin_data_type(%{data_type: t1} = p1, %{data_type: t2} = p2) do
    if t1 != t2 do
      compile_error("The pins #{p1.id} and #{p2.id} has different types")
    end
  end
end

defimpl Cuda.Graph.Factory, for: Cuda.Graph do
  require Cuda
  alias Cuda.Graph.Node

  @doc """
  Creates new graph node
  """
  def new(_, id, module, opts \\ [], env \\ []) do
    proto = Node.proto(module, opts, env)
    graph = %Node{}
            |> Cuda.Graph.Factory.new(id, module, opts, env)
            |> Map.from_struct
    graph = struct(proto, graph)
    graph = case function_exported?(module, :__graph__, 3) do
      true -> module.__graph__(graph, opts, env)
      _    -> graph
    end
    # graph = graph |> Graph.expand()
    graph
  end
end
