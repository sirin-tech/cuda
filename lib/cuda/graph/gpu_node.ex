defmodule Cuda.Graph.GPUNode do
  @moduledoc """
  Represents a graph node that will be executed on GPU.

  ```
  defmodule MyNode do
    use Cuda.Graph.GPUNode

    def __pins__(_assigns) do
      [input(:in), output(:out)]
    end

    def __ptx__(_assigns) do
      \"\"\"
      some ptx code
      \"\"\"
    end
  end
  ```
  """

  alias Cuda.Graph
  alias Cuda.Node
  alias Cuda.Graph.Pin
  alias Cuda.Graph.NodeProto

  require Cuda

  @type t :: %__MODULE__{
    id: Graph.id,
    module: module,
    type: Node.type,
    pins: [Pin.t],
    assigns: map
  }
  @type source :: String.t | [String.t] | nil

  @callback __ptx__(node :: struct) :: source
  @callback __c__(node :: struct) :: source
  @callback __batch__(node :: struct) :: [any]

  @derive [NodeProto]
  defstruct [:id, :module, :type, pins: [], assigns: %{}]

  defmacro __using__(_opts) do
    quote do
      use Cuda.Graph.Node
      @behaviour unquote(__MODULE__)
      def __ptx__(_node), do: []
      def __c__(_node), do: []
      def __batch__(_node), do: []
      def __proto__(), do: unquote(__MODULE__)
      def __type__(_assigns), do: :gpu
      defoverridable __batch__: 1, __c__: 1, __ptx__: 1
    end
  end
end

defimpl Cuda.Graph.Factory, for: Cuda.Graph.GPUNode do
  alias Cuda.Graph.Node
  alias Cuda.Graph.Factory

  @doc """
  Creates a new gpu node
  """
  def new(_, id, module, opts, env) do
    node = %Node{}
           |> Factory.new(id, module, opts, env)
           |> Map.from_struct
    module
    |> Node.proto()
    |> struct(node)
  end
end
