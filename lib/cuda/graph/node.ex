defmodule Cuda.Graph.Node do
  @moduledoc """
  Represents an evaluation graph node.

  You can use this module to define your own evaluation nodes. To do this you
  should implement callbacks that will be called with user options, specified
  at node creation time and current Cuda environment. Here is simple example:

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

  alias Cuda.Graph
  alias Cuda.Graph.Pin

  @type type :: :gpu | :host | :virtual | :graph
  @type t :: %__MODULE__{
    id: Graph.id,
    module: module,
    type: type,
    pins: [Pin.t]
  }

  @doc """
  Retrieves a complete pin list for newly created node.

  You can use `pin/3`, `input/2`, `output/2`, `consumer/2` and `producer/2`
  helpers here.
  """
  @callback __pins__(opts :: keyword, env :: keyword) :: [Pin.t]

  @doc """
  Retrieves a node type.

  Following types are supported:

  * `:virtual` - node does not involved in real computations (it does not change
                 data and does not affect the computation flow). It can be
                 usefull for intermediate data retrieving and so on.
  * `:host`    - node makes host (CPU) computations but does not affects any GPU
                 workflow
  * `:gpu`     - node affects GPU and optionally CPU workflows.
  """
  @callback __type__(opts :: keyword, env :: keyword) :: type

  @types ~w(gpu host virtual graph)a
  @reserved_names ~w(input output)a
  @exports [consumer: 2, input: 2, output: 2, pin: 3, producer: 2]

  defstruct [:id, :module, :type, pins: []]

  defmacro __using__(_opts) do
    quote do
      import unquote(__MODULE__), only: unquote(@exports)
      @behaviour unquote(__MODULE__)
    end
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
  Creates a new evaluation node
  """
  @spec new(module :: module, options :: keyword, env :: keyword) :: t
  def new(module, opts \\ [], env \\ []) do
    id = Keyword.get(opts, :name, Graph.gen_id())
    if id in @reserved_names do
      raise CompileError, description: "Reserved node name '#{id}' used"
    end

    type = case function_exported?(module, :__type__, 2) do
      true -> module.__type__(opts, env)
      _    -> :virtual
    end
    if not type in @types do
      raise CompileError, description: "Unsupported type: #{inspect type}"
    end

    pins = case function_exported?(module, :__pins__, 2) do
      true -> module.__pins__(opts, env)
      _    -> []
    end
    if not is_list(pins) or not Enum.all?(pins, &valid_pin?/1) do
      raise CompileError, description: "Invalid connector list supploed"
    end

    struct(__MODULE__, id: id, module: module, type: type, pins: pins)
  end

  @doc false
  # Returns connector by its id
  @spec get_pin(node:: t, id: Graph.id) :: Pin.t | nil
  def get_pin(%{pins: pins}, id) do
    pins |> Enum.find(fn
      %Pin{id: ^id} -> true
      _                   -> false
    end)
  end
  def get_pin(_, _), do: nil

  @doc false
  # Returns a list of pins of specified type
  @spec get_pins(node :: t, type :: Pin.type) :: Pin.t | nil
  def get_pins(%{pins: pins}, type) do
    pins |> Enum.filter(fn
      %Pin{type: ^type} -> true
      _                       -> false
    end)
  end
  def pins(_, _), do: nil

  defp valid_pin?(%Pin{}), do: true
  defp valid_pin?(_), do: false
end
