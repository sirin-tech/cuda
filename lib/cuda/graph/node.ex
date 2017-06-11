defmodule Cuda.Graph.Node do
  @moduledoc """
  Represents an evaluation graph node.

  You can use this module to define your own evaluation nodes. To do this you
  should implement callbacks that will be called with user options, specified
  at node creation time and current Cuda environment. Here is a simple example:

  ```
  defmodule MyNode do
    use Cuda.Graph.Node

    def __pins__(_assigns) do
      [input(:in), output(:out)]
    end

    def __type__(_assigns) do
      :host
    end
  end
  ```
  """

  require Cuda
  alias Cuda.Graph
  alias Cuda.Graph.Pin
  alias Cuda.Graph.NodeProto

  @type type :: :gpu | :host | :virtual | :graph | :computation_graph
  @type options :: keyword
  @type assigns :: %{options: options, env: Cuda.Env.t}
  @type t :: %__MODULE__{
    id: Graph.id,
    module: module,
    type: type,
    pins: [Pin.t],
    assigns: assigns
  }

  @callback __assigns__(id :: Graph.id, opts :: options, env :: Cuda.Env.t) :: map | keyword

  @doc """
  Provides a node protocol that is a structurethat holds node data.

  It can be for example `Cuda.Graph`, `Cuda.Graph.Node`, `Cuda.Graph.GPUNode`
  or any other module that implements node protocol functionality.

  By default it will be `Cuda.Graph.Node`.
  """
  @callback __proto__() :: atom

  @doc """
  Provides a complete pin list for newly created node.

  You can use `pin/3`, `input/2`, `output/2`, `consumer/2` and `producer/2`
  helpers here.
  """
  @callback __pins__(assigns :: assigns) :: [Pin.t]

  @doc """
  Provides a node type.

  Following types are supported:

  * `:virtual` - node does not involved in real computations (it does not change
                 data and does not affect the computation flow). It can be
                 usefull for intermediate data retrieving and so on.
  * `:host`    - node makes host (CPU) computations but does not affects any GPU
                 workflow
  * `:gpu`     - node affects GPU and optionally CPU workflows
  * `:graph`   - node with graph nested in it
  """
  @callback __type__(assigns :: assigns) :: type

  @doc """
  Called before compilation.

  You can put vars, helpers and other stuff needed by further compilation
  process.
  """
  @callback __compile__(node :: struct) :: {:ok, struct} | {:error, any}

  @derive [NodeProto]
  defstruct [:id, :module, :type, pins: [], assigns: %{}]

  @exports [consumer: 2, consumer: 3, input: 2, input: 3, output: 2, output: 3,
            pin: 3, pin: 4, producer: 2, producer: 3]
  @input_pins  ~w(input consumer terminator)a
  @output_pins ~w(output producer)a
  @graph_types ~w(graph computation_graph)a

  defmacro __using__(_opts) do
    quote do
      import unquote(__MODULE__), only: unquote(@exports)
      import Cuda.Graph.NodeProto, only: [assign: 3]
      @behaviour unquote(__MODULE__)
      def __assigns__(_id, _opts, _env), do: %{}
      def __proto__(), do: unquote(__MODULE__)
      def __compile__(node), do: {:ok, node}
      defoverridable __assigns__: 3, __compile__: 1, __proto__: 0
    end
  end

  defmacro input_pin_types() do
    quote(do: unquote(@input_pins))
  end

  defmacro output_pin_types() do
    quote(do: unquote(@output_pins))
  end

  defmacro graph_types() do
    quote(do: unquote(@graph_types))
  end

  @doc """
  Creates a pin with specified parameters
  """
  @spec pin(name :: Graph.id, type :: Pin.type, data_type :: any) :: Pin.t
  @spec pin(name :: Graph.id, type :: Pin.type, data_type :: any, group :: Pin.group) :: Pin.t
  def pin(name, type, data_type, group \\ nil) do
    %Pin{
      id: name,
      type: type,
      data_type: data_type,
      group: group
    }
  end

  @doc """
  Creates an input pin with specified parameters.

  Input is a pin from which the data passed inside an evaluation node.
  """
  @spec input(name :: Graph.id, data_type :: any) :: Pin.t
  @spec input(name :: Graph.id, data_type :: any, group :: Pin.group) :: Pin.t
  def input(name, data_type, group \\ nil) do
    pin(name, :input, data_type, group)
  end

  @doc """
  Creates an output pin with specified parameters.

  Ouput is a pin through which you pass data outside from your node.
  """
  @spec output(name :: Graph.id, data_type :: any) :: Pin.t
  @spec output(name :: Graph.id, data_type :: any, group :: Pin.group) :: Pin.t
  def output(name, data_type, group \\ nil) do
    pin(name, :output, data_type, group)
  end

  @doc """
  Creates a producer pin with specified parameters.

  Producers are nodes that generates some data. Data from this kind of pin can
  be passed to `:input` or `:consumer` pins.
  """
  @spec producer(name :: Graph.id, data_type :: any) :: Pin.t
  @spec producer(name :: Graph.id, data_type :: any, group :: Pin.group) :: Pin.t
  def producer(name, data_type, group \\ nil) do
    pin(name, :producer, data_type, group)
  end

  @doc """
  Creates a consumer pin with specified parameters.

  Consumers are nodes that takes some data. This pin is like a data flow
  terminator. Data for this pin can be taked from `:output` or `:producer`
  pins.
  """
  @spec consumer(name :: Graph.id, data_type :: any) :: Pin.t
  @spec consumer(name :: Graph.id, data_type :: any, group :: Pin.group) :: Pin.t
  def consumer(name, data_type, group \\ nil) do
    pin(name, :consumer, data_type, group)
  end

  @doc """
  Returns module of struct that used to store node data. It can be for example
  `Cuda.Graph`, `Cuda.Graph.Node`, `Cuda.Graph.GPUNode` or any other module,
  related to node type.
  """
  @spec proto(module :: atom) :: atom
  def proto(module) do
    if function_exported?(module, :__proto__, 0) do
      module.__proto__()
    else
      __MODULE__
    end
  end

  def string_id(id) when is_tuple(id) do
    id |> Tuple.to_list |> Enum.map(&string_id/1) |> Enum.join("__")
  end
  def string_id(id) do
    "#{id}"
  end
end

defimpl Cuda.Graph.Factory, for: Cuda.Graph.Node do
  require Cuda
  alias Cuda.Graph.Pin

  @types ~w(gpu host virtual graph computation_graph)a
  @reserved_names ~w(input output)a

  def new(_, id, module, opts, env) do
    with {:module, module} <- Code.ensure_loaded(module) do
      if id in @reserved_names do
        Cuda.compile_error("Reserved node name '#{id}' used")
      end

      assigns = case function_exported?(module, :__assigns__, 3) do
        true -> module.__assigns__(id, opts, env) |> Enum.into(%{})
        _    -> %{}
      end
      assigns = Map.merge(assigns, %{options: opts, env: env})

      type = case function_exported?(module, :__type__, 1) do
        true -> module.__type__(assigns)
        _    -> :virtual
      end
      if not type in @types do
        Cuda.compile_error("Unsupported type: #{inspect type}")
      end

      pins = case function_exported?(module, :__pins__, 1) do
        true -> module.__pins__(assigns)
        _    -> []
      end
      if not is_list(pins) or not Enum.all?(pins, &valid_pin?/1) do
        Cuda.compile_error("Invalid pin list supplied")
      end

      struct(Cuda.Graph.Node, id: id, module: module, type: type, pins: pins,
                              assigns: assigns)
    else
      _ -> Cuda.compile_error("Node module #{module} could not be loaded")
    end
  end

  defp valid_pin?(%Pin{}), do: true
  defp valid_pin?(_), do: false
end
