defmodule Cuda.Graph.Node do
  @moduledoc """
  Represents an evaluation graph node.

  You can use this module to define your own evaluation nodes. To do this you
  should implement callbacks that will be called with user options, specified
  at node creation time and current Cuda environment. Here is a simple example:

  ```
  defmodule MyNode do
    use Cuda.Graph.Node

    def __pins__(_opts, _env) do
      [input(:in), output(:out)]
    end

    def __type__(_opts, _env) do
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
  @type t :: %__MODULE__{
    id: Graph.id,
    module: module,
    type: type,
    pins: [Pin.t],
    assigns: map
  }

  @doc """
  Provides a node protocol that is a structurethat holds node data.

  It can be for example `Cuda.Graph`, `Cuda.Graph.Node`, `Cuda.Graph.GPUNode`
  or any other module that implements node protocol functionality.

  By default it will be `Cuda.Graph.Node`.
  """
  @callback __proto__(opts :: options, env :: Cuda.Env.t) :: atom

  @doc """
  Provides a complete pin list for newly created node.

  You can use `pin/3`, `input/2`, `output/2`, `consumer/2` and `producer/2`
  helpers here.
  """
  @callback __pins__(opts :: options, env :: Cuda.Env.t) :: [Pin.t]

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
  @callback __type__(opts :: options, env :: Cuda.Env.t) :: type

  @derive [NodeProto]
  defstruct [:id, :module, :type, pins: [], assigns: %{}]

  @exports [consumer: 2, input: 2, output: 2, pin: 3, producer: 2]
  @input_pins  ~w(input consumer)a
  @output_pins ~w(output producer)a
  @graph_types ~w(graph computation_graph)a

  defmacro __using__(_opts) do
    quote do
      import unquote(__MODULE__), only: unquote(@exports)
      @behaviour unquote(__MODULE__)
      def __proto__(_opts, _env), do: unquote(__MODULE__)
      defoverridable __proto__: 2
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
  def pin(name, type, data_type) do
    %Pin{
      id: name,
      type: type,
      data_type: data_type
    }
  end

  @doc """
  Creates an input pin with specified parameters.

  Input is a pin from which the data passed inside an evaluation node.
  """
  @spec input(name :: Graph.id, data_type :: any) :: Pin.t
  def input(name, data_type), do: pin(name, :input, data_type)

  @doc """
  Creates an output pin with specified parameters.

  Ouput is a pin through which you pass data outside from your node.
  """
  @spec output(name :: Graph.id, data_type :: any) :: Pin.t
  def output(name, data_type), do: pin(name, :output, data_type)

  @doc """
  Creates a producer pin with specified parameters.

  Producers are nodes that generates some data. Data from this kind of pin can
  be passed to `:input` or `:consumer` pins.
  """
  @spec producer(name :: Graph.id, data_type :: any) :: Pin.t
  def producer(name, data_type), do: pin(name, :producer, data_type)

  @doc """
  Creates a consumer pin with specified parameters.

  Consumers are nodes that takes some data. This pin is like a data flow
  terminator. Data for this pin can be taked from `:output` or `:producer`
  pins.
  """
  @spec consumer(name :: Graph.id, data_type :: any) :: Pin.t
  def consumer(name, data_type), do: pin(name, :consumer, data_type)

  @doc """
  Returns module of struct that used to store node data. It can be for example
  `Cuda.Graph`, `Cuda.Graph.Node`, `Cuda.Graph.GPUNode` or any other module,
  related to node type.
  """
  @spec proto(module :: atom, opts :: keyword, env :: Cuda.Env.t) :: atom
  def proto(module, opts, env) do
    if function_exported?(module, :__proto__, 2) do
      module.__proto__(opts, env)
    else
      __MODULE__
    end
  end
end

defimpl Cuda.Graph.Factory, for: Cuda.Graph.Node do
  require Cuda
  alias Cuda.Graph.Pin

  @types ~w(gpu host virtual graph computation_graph)a
  @reserved_names ~w(input output)a

  def new(_, id, module, opts \\ [], env \\ []) do
    with {:module, module} <- Code.ensure_loaded(module) do
      if id in @reserved_names do
        Cuda.compile_error("Reserved node name '#{id}' used")
      end

      type = case function_exported?(module, :__type__, 2) do
        true -> module.__type__(opts, env)
        _    -> :virtual
      end
      if not type in @types do
        Cuda.compile_error("Unsupported type: #{inspect type}")
      end

      pins = case function_exported?(module, :__pins__, 2) do
        true -> module.__pins__(opts, env)
        _    -> []
      end
      if not is_list(pins) or not Enum.all?(pins, &valid_pin?/1) do
        Cuda.compile_error("Invalid pin list supplied")
      end

      struct(Cuda.Graph.Node, id: id, module: module, type: type, pins: pins,
                              assigns: %{options: opts})
    else
      _ -> Cuda.compile_error("Node module #{module} could not be loaded")
    end
  end

  defp valid_pin?(%Pin{}), do: true
  defp valid_pin?(_), do: false
end
