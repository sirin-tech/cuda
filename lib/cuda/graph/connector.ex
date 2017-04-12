defmodule Cuda.Graph.Connector do
  alias Cuda.Graph

  @type type :: :input | :output | :producer | :consumer

  @type t :: %__MODULE__{
    id: Graph.id,
    type: type,
    data_type: any
  }

  defstruct [:id, :type, :data_type]
end
