defmodule Cuda.Graph.Pin do
  @moduledoc """
  Represents evaluation graph node connector.
  """

  alias Cuda.Graph

  @type type :: :input | :output | :producer | :consumer

  @type t :: %__MODULE__{
    id: Graph.id,
    type: type,
    data_type: any
  }

  defstruct [:id, :type, :data_type]
end
