defmodule Cuda.Graph.Pin do
  @moduledoc """
  Represents evaluation graph node connector (pin).
  """

  alias Cuda.Graph

  @type type :: :input | :output | :producer | :consumer

  @type t :: %__MODULE__{
    id: Graph.id,
    type: type,
    data_type: any
  }

  defstruct [:id, :type, :data_type]

  def data_size(%__MODULE__{data_type: data_type}) do
    type_size(data_type)
  end
  def data_size(_), do: 0

  @type_re ~r/(\d+)/
  defp type_size(:i8),  do: 1
  defp type_size(:i16), do: 2
  defp type_size(:i32), do: 4
  defp type_size(:i64), do: 8
  defp type_size(:u8),  do: 1
  defp type_size(:u16), do: 2
  defp type_size(:u32), do: 4
  defp type_size(:u64), do: 8
  defp type_size(:f16), do: 2
  defp type_size(:f32), do: 4
  defp type_size(:f64), do: 8
  defp type_size(type) when is_atom(type) do
    case Regex.run(@type_re, "#{type}", capture: :all_but_first) do
      [n] -> div(String.to_integer(n), 8)
      _   -> 0
    end
  end
  defp type_size(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list
    |> Enum.map(&type_size/1)
    |> Enum.reduce(1, &Kernel.*/2)
  end
  defp type_size(_), do: 0
end
