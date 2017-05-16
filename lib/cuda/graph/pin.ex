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

  def pack(x, :i8), do: <<x>>
  def pack(x, :i16), do: <<x::little-size(16)>>
  def pack(x, :i32), do: <<x::little-size(32)>>
  def pack(x, :i64), do: <<x::little-size(64)>>
  def pack(x, "i8"), do: pack(x, :i8)
  def pack(x, "i16"), do: pack(x, :i16)
  def pack(x, "i32"), do: pack(x, :i32)
  def pack(x, "i64"), do: pack(x, :i64)
  def pack(x, :u8), do: <<x::unsigned>>
  def pack(x, :u16), do: <<x::unsigned-little-size(16)>>
  def pack(x, :u32), do: <<x::unsigned-little-size(32)>>
  def pack(x, :u64), do: <<x::unsigned-little-size(64)>>
  def pack(x, "u8"), do: pack(x, :u8)
  def pack(x, "u16"), do: pack(x, :u16)
  def pack(x, "u32"), do: pack(x, :u32)
  def pack(x, "u64"), do: pack(x, :u64)
  # TODO: pack 16-bit floats
  # def pack(x, :f16), do: <<x::float-little-size(16)>>
  def pack(x, :f32), do: <<x::float-little-size(32)>>
  def pack(x, :f64), do: <<x::float-little-size(64)>>
  # def pack(x, "f16"), do: pack(x, :f16)
  def pack(x, "f32"), do: pack(x, :f32)
  def pack(x, "f64"), do: pack(x, :f64)
  def pack(x, {type, arity}) when not is_tuple(arity), do: pack(x, {type, {arity}})
  def pack(x, {type, arity}) when is_list(x) do
    arity = type_size(arity)
    x = List.flatten(x)
    if length(x) == arity do
      x |> Enum.map(& pack(&1, type)) |> Enum.join
    else
      <<>>
    end
  end
  def pack(x, types) when is_list(types) and is_list(x) and length(types) == length(x) do
    x
    |> Enum.zip(types)
    |> Enum.map(fn {x, type} -> pack(x, type) end)
    |> Enum.join()
  end
  def pack(_, _), do: <<>>

  def unpack(<<x>>, :i8), do: x
  def unpack(<<x::little-size(16)>>, :i16), do: x
  def unpack(<<x::little-size(32)>>, :i32), do: x
  def unpack(<<x::little-size(64)>>, :i64), do: x
  def unpack(x, "i8"),  do: unpack(x, :i8)
  def unpack(x, "i16"), do: unpack(x, :i16)
  def unpack(x, "i32"), do: unpack(x, :i32)
  def unpack(x, "i64"), do: unpack(x, :i64)
  def unpack(<<x::unsigned>>, :u8), do: x
  def unpack(<<x::unsigned-little-size(16)>>, :u16), do: x
  def unpack(<<x::unsigned-little-size(32)>>, :u32), do: x
  def unpack(<<x::unsigned-little-size(64)>>, :u64), do: x
  def unpack(x, "u8"),  do: unpack(x, :u8)
  def unpack(x, "u16"), do: unpack(x, :u16)
  def unpack(x, "u32"), do: unpack(x, :u32)
  def unpack(x, "u64"), do: unpack(x, :u64)
  # TODO: pack 16-bit floats
  # def pack(x, :f16), do: <<x::float-little-size(16)>>
  def unpack(<<x::float-little-size(32)>>, :f32), do: x
  def unpack(<<x::float-little-size(64)>>, :f64), do: x
  # def pack(x, "f16"), do: pack(x, :f16)
  def unpack(x, "f32"), do: unpack(x, :f32)
  def unpack(x, "f64"), do: unpack(x, :f64)
  def unpack(x, {type, arity}) when is_tuple(arity) do
    {list, _} = unpack_list(x, {type, Tuple.to_list(arity)})
    list
  end
  def unpack(x, {type, arity}) when not is_tuple(arity) do
    {list, _} = unpack_list(x, {type, [arity]})
    list
  end
  def unpack(x, types) when is_list(types) do
    types |> Enum.reduce({[], x}, fn
      type, {list, rest} ->
        {data, rest} = unpack_list(rest, type)
        {list ++ data, rest}
      _, acc ->
        acc
    end)
  end
  def unpack(_, _), do: nil

  defp unpack_list(x, {type, [arity]}) do
    size = type_size(type)
    Enum.reduce(1..arity, {[], x}, fn
      _, {list, <<x::unit(8)-size(size), rest::binary>>} ->
        {list ++ [unpack(x, type)], rest}
      _, acc ->
        acc
    end)
  end
  defp unpack_list(x, {type, [current | arity]}) do
    Enum.reduce(1..current, {[], x}, fn
      _, {list, rest} ->
        {data, rest} = unpack_list(rest, {type, arity})
        {list ++ [data], rest}
      _, acc ->
        acc
    end)
  end
  defp unpack_list(x, type) do
    size = type_size(type)
    <<x::unit(8)-size(size), rest::binary>> = x
    {unpack(x, type), rest}
  end

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
  defp type_size(type) when is_atom(type) or is_bitstring(type) do
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
  defp type_size(i) when is_integer(i), do: i
  defp type_size(_), do: 0
end
