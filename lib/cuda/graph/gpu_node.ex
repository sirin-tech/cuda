defmodule Cuda.Graph.GPUNode do
  @moduledoc """
  Represents a graph node that will be executed on GPU.

  ```
  defmodule MyNode do
    use Cuda.Graph.GPUNode

    def __pins__(_opts, _env) do
      [input(:in), output(:out)]
    end

    def __ptx__(_opts, _ctx) do
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
  alias Cuda.Compiler.Context

  require Cuda

  @type t :: %__MODULE__{
    id: Graph.id,
    module: module,
    type: Node.type,
    pins: [Pin.t],
    assigns: map
  }
  @type source :: String.t | [String.t] | nil

  @callback __ptx__(opts :: Node.options, ctx :: Context.t) :: source
  @callback __c__(opts :: Node.options, ctx :: Context.t) :: source
  @callback __vars__(opts :: Node.options, ctx :: Context.t) :: keyword | map
  @callback __helpers__(opts :: Node.options, ctx :: Context.t) :: [atom]

  @derive [NodeProto]
  defstruct [:id, :module, :type, pins: [], assigns: %{}]

  defmacro __using__(_opts) do
    quote do
      use Cuda.Graph.Node
      @behaviour unquote(__MODULE__)
      def __ptx__(_opts, _ctx), do: []
      def __c__(_opts, _ctx), do: []
      def __proto__(_opts, _env), do: unquote(__MODULE__)
      def __type__(_opts, _env), do: :gpu
      def __vars__(_opts, _ctx), do: %{}
      def __helpers__(_opts, _ctx), do: []
      defoverridable __c__: 2, __helpers__: 2, __ptx__: 2, __vars__: 2
    end
  end
end

defimpl Cuda.Graph.Factory, for: Cuda.Graph.GPUNode do
  alias Cuda.Graph.Node

  @doc """
  Creates a new gpu node
  """
  def new(_, id, module, opts \\ [], env \\ []) do
    node = %Node{}
           |> Cuda.Graph.Factory.new(id, module, opts, env)
           |> Map.from_struct
    module
    |> Node.proto(opts, env)
    |> struct(node)
  end
end
