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
  @type link_spec :: {id, id}
  @type link :: {link, link}

  @type t :: %__MODULE__{
    id: id,
    module: module,
    type: Node.type,
    pins: [Pin.t],
    nodes: [Node.t],
    links: [link],
    assigns: map
  }

  @type dfs_action :: :enter | :move | :leave
  @type dfs_result :: {:ok | :error | atom, state :: any}
  @type dfs_callback :: (action :: dfs_action, arg :: any, state :: any -> dfs_result)

  @callback __graph__(graph :: t) :: t
  @callback __child_options__(id :: id, module :: atom, graph :: t) :: Node.options

  @derive [NodeProto, GraphProto]
  defstruct [:id, :module, type: :graph, pins: [], nodes: [], links: [],
             assigns: %{}]

  import Node, only: [input_pin_types: 0, output_pin_types: 0]

  @self :__self__
  @exports [add: 3, add: 4,
            chain: 3, chain: 4,
            close: 1, link: 3]

  defmacro __using__(_opts) do
    quote do
      use Cuda.Graph.Node
      import unquote(__MODULE__), only: unquote(@exports)
      @behaviour unquote(__MODULE__)
      def __type__(_assigns), do: :graph
      def __proto__(), do: unquote(__MODULE__)
      def __child_options__(_id, _module, _graph), do: []

      defoverridable __child_options__: 3, __type__: 1
    end
  end

  def add(%__MODULE__{} = graph, id, module, opts \\ []) do
    with {:module, module} <- Code.ensure_loaded(module) do
      if id == graph.id do
        compile_error("The id `#{id}` of newly added node is already used by graph")
      end
      if GraphProto.node(graph, id) != nil do
        compile_error("Node with id `#{id}` s already exists in the graph")
      end
      proto = struct(Node.proto(module))
      opts = case function_exported?(graph.module, :__child_options__, 3) do
        true -> id
                |> graph.module.__child_options__(module, graph)
                |> Keyword.merge(opts)
        _    -> opts
      end
      env = graph |> Map.get(:assigns, %{}) |> Map.get(:env, %Cuda.Env{})
      GraphProto.add(graph, Cuda.Graph.Factory.new(proto, id, module, opts, env))
    else
      _ -> compile_error("Graph module #{module} could not be loaded")
    end
  end

  def chain(graph, id, module, opts \\ [])
  def chain(%__MODULE__{nodes: []} = graph, id, module, opts) do
    src_pin = case NodeProto.pins(graph, input_pin_types()) do
      [src_pin] -> src_pin
      _         -> compile_error("Chain allowed only for graphs with single input")
    end
    with %{nodes: [node]} = graph <- add(graph, id, module, opts) do
      dst_pin = case NodeProto.pins(node, input_pin_types()) do
        [dst_pin] -> dst_pin
        _         -> compile_error("Chain can only be applied to nodes with single input")
      end
      link(graph, src_pin.id, {id, dst_pin.id})
    end
  end
  def chain(%__MODULE__{nodes: [src_node | _]} = graph, id, module, opts) do
    src_pin = case NodeProto.pins(src_node, output_pin_types()) do
      [src_pin] -> src_pin
      _         -> compile_error("Chain can only be applied after nodes with single output")
    end
    with %{nodes: [dst_node | _]} = graph <- add(graph, id, module, opts) do
      dst_pin = case NodeProto.pins(dst_node, input_pin_types()) do
        [dst_pin] -> dst_pin
        _         -> compile_error("Chain can only be applied to nodes with single input")
      end
      link(graph, {src_node.id, src_pin.id}, {dst_node.id, dst_pin.id})
    end
  end

  def close(%__MODULE__{nodes: []} = graph) do
    src_pin = case NodeProto.pins(graph, input_pin_types()) do
      [src_pin] -> src_pin
      _         -> compile_error("Close allowed only for graphs with single input")
    end
    dst_pin = case NodeProto.pins(graph, output_pin_types()) do
      [dst_pin] -> dst_pin
      _         -> compile_error("Close allowed only for graphs with single output")
    end
    link(graph, src_pin.id, dst_pin.id)
  end
  def close(%__MODULE__{nodes: [node | _]} = graph) do
    src_pin = case NodeProto.pins(node, output_pin_types()) do
      [src_pin] -> src_pin
      _         -> compile_error("Close can only be applied after nodes with single output")
    end
    dst_pin = case NodeProto.pins(graph, output_pin_types()) do
      [dst_pin] -> dst_pin
      _         -> compile_error("Close allowed only for graphs with single output")
    end
    link(graph, {node.id, src_pin.id}, dst_pin.id)
  end

  def link(%__MODULE__{links: links} = graph, {sn, sp} = src, {dn, dp} = dst) do
    # node to node connection
    with {:src, %{} = src_node} <- {:src, GraphProto.node(graph, sn)},
         {:dst, %{} = dst_node} <- {:dst, GraphProto.node(graph, dn)} do
      #IO.inspect({src, dst})
      src_pin = assert_pin_type(src_node, sp, output_pin_types())
      dst_pin = assert_pin_type(dst_node, dp, input_pin_types())
      assert_pin_data_type(src_pin, dst_pin)
      assert_pin_layout(src_pin, dst_pin)
      %{graph | links: [{src, dst} | links]}
    else
      {:src, _} -> compile_error("Source node `#{sn}` not found")
      {:dst, _} -> compile_error("Destination node `#{dn}` not found")
    end
  end

  def link(%__MODULE__{links: links} = graph, src, {dn, dp} = dst) do
    # input to node connection
    with %{} = dst_node <- GraphProto.node(graph, dn) do
      #IO.inspect({graph.id, src, dst})
      src_pin = assert_pin_type(graph, src, input_pin_types())
      dst_pin = assert_pin_type(dst_node, dp, input_pin_types())
      assert_pin_data_type(src_pin, dst_pin)
      assert_pin_layout(src_pin, dst_pin)
      %{graph | links: [{{@self, src}, dst} | links]}
    else
      _ -> compile_error("Destination node `#{dn}` not found")
    end
  end

  def link(%__MODULE__{links: links} = graph, {sn, sp} = src, dst) do
    # node to output connection
    with %{} = src_node <- GraphProto.node(graph, sn) do
      src_pin = assert_pin_type(graph, dst, output_pin_types())
      dst_pin = assert_pin_type(src_node, sp, output_pin_types())
      assert_pin_data_type(src_pin, dst_pin)
      assert_pin_layout(src_pin, dst_pin)
      %{graph | links: [{src, {@self, dst}} | links]}
    else
      _ -> compile_error("Source node `#{sn}` not found")
    end
  end

  def link(%__MODULE__{links: links} = graph, src, dst) do
    # input to output connection
    src_pin = assert_pin_type(graph, src, input_pin_types())
    dst_pin = assert_pin_type(graph, dst, output_pin_types())
    assert_pin_data_type(src_pin, dst_pin)
    assert_pin_layout(src_pin, dst_pin)
    %{graph | links: [{{@self, src}, {@self, dst}} | links]}
  end

  defp assert_pin_type(node, pin_name, types) do
    with %Pin{} = pin <- NodeProto.pin(node, pin_name) do
      if not pin.type in types do
        types = types |> Enum.map(& "#{&1}") |> Enum.join(" or ")
        id = Node.string_id(node.id)
        compile_error("Pin `#{pin_name}` of node `#{id}` has a wrong" <>
                      " type. The #{types} types are expected.")
      end
      pin
    else
      _ ->
        id = Node.string_id(node.id)
        compile_error("Pin `#{pin_name}` not found in node `#{id}`")
    end
  end

  # TODO: move pin type checking logic into Cuda.Graph.Pin
  defp assert_pin_data_type(%{data_type: {t1, a1}} = p1, %{data_type: {t2, a2}} = p2) do
    s1 = Pin.data_size(a1)
    s2 = Pin.data_size(a2)
    if t1 != t2 or s1 != s2 do
      compile_error("The pins #{p1.id} and #{p2.id} has different types")
    end
  end
  # nil data_type assumes auto-detection
  defp assert_pin_data_type(%{data_type: nil}, _), do: true
  defp assert_pin_data_type(_, %{data_type: nil}), do: true
  defp assert_pin_data_type(%{data_type: t1} = p1, %{data_type: t2} = p2) do
    if t1 != t2 do
      compile_error("The pins #{p1.id} and #{p2.id} has different types")
    end
  end

  defp assert_pin_layout(%{id: id1, layout: l1}, %{id: id2, layout: l2}) when l1 != l2 do
    compile_error("The pins #{id1} and #{id2} has different layout")
  end
  defp assert_pin_layout(_, _), do: true
end

defimpl Cuda.Graph.Factory, for: Cuda.Graph do
  require Cuda
  alias Cuda.Graph.{Node, NodeProto}

  @doc """
  Creates new graph node
  """
  def new(_, id, module, opts, env) do
    with {:module, module} <- Code.ensure_loaded(module) do
      proto = Node.proto(module)
      graph = %Node{}
              |> Cuda.Graph.Factory.new(id, module, opts, env)
              |> Map.from_struct
      graph = struct(proto, graph)
      graph = case function_exported?(module, :__graph__, 1) do
        true -> module.__graph__(graph)
        _    -> graph
      end
      graph |> set_pin_shapes()
    else
      _ -> Cuda.compile_error("Node module #{module} could not be loaded")
    end
  end

  defp set_pin_shapes(%{pins: pins} = graph) do
    %{graph | pins: pins |> Enum.map(& set_pin_shape(graph, &1))}
  end

  defp set_pin_shape(graph, %{alias: {:group, a}, data_type: nil} = pin) when not is_nil(a) do
    pins = graph.nodes
           |> Enum.map(fn node ->
             pins = node.pins
                    |> Enum.filter(& &1.group == a)
                    |> Enum.map(& {&1.id, &1.data_type})
                    |> Enum.into(%{})
             {node.id, pins}
           end)
           |> Enum.filter(fn
             {_, []} -> false
             _       -> true
           end)
           |> Enum.into(%{})
    case pins do
      []   -> Cuda.compile_error("Invalid pin alias group: #{inspect a}")
      pins -> %{pin | data_type: pins}
    end
  end
  defp set_pin_shape(graph, %{alias: a, data_type: nil} = pin) when not is_nil(a) do
    case NodeProto.pin(graph, a) do
      nil     -> Cuda.compile_error("Invalid pin alias: #{inspect a}")
      aliases -> %{pin | data_type: aliases.data_type}
    end
  end
  defp set_pin_shape(_, pin), do: pin
end
