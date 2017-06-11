defmodule Cuda.Graph.Pin do
  @moduledoc """
  Represents evaluation graph node connector (pin).
  """

  alias Cuda.Graph
  require Logger

  @type type :: :input | :output | :producer | :consumer | :terminator
  @type group :: nil | atom
  @type alias :: nil | {:group, atom} | Graph.id

  @type t :: %__MODULE__{
    id: Graph.id,
    type: type,
    group: group,
    alias: alias,
    data_type: any
  }

  defstruct [:id, :type, :group, :alias, :data_type]

  def data_size(%__MODULE__{data_type: data_type}) do
    type_size(data_type)
  end
  def data_size(_), do: 0

  def data_arity(%__MODULE__{data_type: {_, arity}}) do
    type_size(arity)
  end
  def data_arity(%__MODULE__{data_type: _}), do: 1
  def data_arity(_), do: 0

  def data_type(%__MODULE__{data_type: {t, _}}), do: t
  def data_type(%__MODULE__{data_type: t}), do: t
  def data_type(_), do: nil

  def pack(:zero, t) when is_atom(t) or is_bitstring(t), do: pack(0, t)
  def pack(_, {:skip, bytes}), do: <<0::unit(8)-size(bytes)>>
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
      raise RuntimeError, message: "Arity of array #{inspect x} should be #{arity}"
    end
  end
  def pack(:zero, {type, arity}) do
    size = type_size(arity) * type_size(type)
    <<0::unit(8)-size(size)>>
  end
  def pack(x, types) when is_list(types) and is_list(x) do
    x = case length(types) - length(x) do
      n when n > 0 -> x ++ List.duplicate(0, n)
      _            -> x
    end
    x
    |> Enum.zip(types)
    |> Enum.map(fn {x, type} -> pack(x, type) end)
    |> Enum.join()
  end
  def pack(:zero, types) when is_list(types) do
    types |> Enum.map(& pack(:zero, &1)) |> Enum.join()
  end
  def pack(x, types) when is_list(types), do: pack([x], types)
  def pack(x, types) when is_map(types) and is_map(x) do
    types
    |> Enum.map(fn {k, type} ->
      with {:ok, v} <- Map.fetch(x, k) do
        pack(v, type)
      else
        _ -> raise RuntimeError, message: "Coudn't find value for key #{k}"
      end
    end)
    |> Enum.join()
  end
  def pack(:zero, types) when is_map(types) do
    types |> Enum.map(fn {_, type} -> pack(:zero, type) end) |> Enum.join()
  end
  def pack(nil, type) do
    Logger.warn("Attempt to pack `nil` value for type #{inspect type}")
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
    arity = arity |> Tuple.to_list |> Enum.reverse
    {list, _} = unpack_list(x, {type, arity})
    list
  end
  def unpack(x, {type, arity}) when not is_tuple(arity) do
    {list, _} = unpack_list(x, {type, [arity]})
    list
  end
  def unpack(x, types) when is_list(types) do
    {list, _} = types |> Enum.reduce({[], x}, fn
      type, {list, rest} ->
        {data, rest} = unpack_list(rest, type)
        {list ++ data, rest}
      _, acc ->
        acc
    end)
    list
  end
  def unpack(x, types) when is_map(types) do
    {list, _} = types |> Enum.reduce({%{}, x}, fn
      {k, type}, {map, rest} ->
        {[data], rest} = unpack_list(rest, type)
        {Map.put(map, k, data), rest}
      _, acc ->
        acc
    end)
    list
  end
  def unpack(_, _), do: nil

  defp unpack_list(x, {type, [arity]}) do
    size = type_size(type)
    Enum.reduce(1..arity, {[], x}, fn
      _, {list, <<x::binary-size(size), rest::binary>>} ->
        data = [unpack(x, type)]
        {list ++ data, rest}
      _, acc ->
        acc
    end)
  end
  defp unpack_list(x, {type, [current | arity]}) do
    Enum.reduce(1..current, {[], x}, fn
      _, {list, rest} ->
        {data, rest} = unpack_list(rest, {type, arity})# |> IO.inspect
        {list ++ [data], rest}
      _, acc ->
        acc
    end)
  end
  defp unpack_list(x, {:skip, bytes}) do
    <<_::binary-size(bytes), rest::binary>> = x
    {[], rest}
  end
  defp unpack_list(x, type) do
    size = type_size(type)
    #IO.inspect({x, type, size, byte_size(x)})
    <<x::binary-size(size), rest::binary>> = x
    {[unpack(x, type)], rest}
  end

  @type_re ~r/(\d+)/
  def type_size(:i8),  do: 1
  def type_size(:i16), do: 2
  def type_size(:i32), do: 4
  def type_size(:i64), do: 8
  def type_size(:u8),  do: 1
  def type_size(:u16), do: 2
  def type_size(:u32), do: 4
  def type_size(:u64), do: 8
  def type_size(:f16), do: 2
  def type_size(:f32), do: 4
  def type_size(:f64), do: 8
  def type_size({:skip, n}), do: n
  def type_size(type) when is_atom(type) or is_bitstring(type) do
    case Regex.run(@type_re, "#{type}", capture: :all_but_first) do
      [n] -> div(String.to_integer(n), 8)
      _   -> 0
    end
  end
  def type_size(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list
    |> Enum.map(&type_size/1)
    |> Enum.reduce(1, &Kernel.*/2)
  end
  def type_size(i) when is_integer(i), do: i
  def type_size(l) when is_list(l) do
    l |> Enum.map(&type_size/1) |> Enum.reduce(0, &+/2)
  end
  def type_size(_), do: 0
end
