defmodule Cuda.Graph.GPUNode do
  @moduledoc """
  Represents a graph node that will be executed on GPU.

  ```
  defmodule MyNode do
    use Cuda.Graph.GPUNode

    def __pins__(_opts, _env) do
      [input(:in), output(:out)]
    end

    def __ptx__(_opts, _env) do
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
    pins: [Pin.t]
  }

  @derive [NodeProto]
  defstruct [:id, :module, :type, pins: []]

  defmacro __using__(_opts) do
    quote do
      use Cuda.Graph.Node
      @behaviour unquote(__MODULE__)
    end
  end
end
