defmodule Cuda.Memory do
  require Logger
  alias Cuda.Float16

  defmodule Shape do
    defstruct [:type, skip: 0]
    def new(x), do: x
  end

  defstruct type: :owned, vars: []

  def new(vars, type \\ :owned) do
    {vars, _} = Enum.reduce(vars, {[], 0}, fn
      {k, {o, _} = v}, {vars, offset} when is_integer(o) ->
        {vars ++ [{k, v}], offset}
      {k, type}, {vars, offset} ->
        {vars ++ [{k, {offset, type}}], offset + size(type)}
    end)
    %__MODULE__{vars: vars, type: type}
  end

  def get(%__MODULE__{vars: vars}, key, default \\ nil) do
    with {_, v} <- get_var(vars, key, default), do: v
  end

  def has_key?(%__MODULE__{vars: vars}, key) do
    Enum.any?(vars, fn
      {^key, _} -> true
      _         -> false
    end)
  end

  defp get_var(vars, key, default) do
    with {_, v} <- Enum.find(vars, fn {k, _} -> k == key end) do
      v
    else
      _ -> default
    end
  end

  def offset(nil, _), do: nil
  def offset(_, nil), do: nil
  def offset(%__MODULE__{vars: vars}, [field | rest]) do
    case Keyword.get(vars, field) do
      {offset, type} ->
        with n when is_integer(n) <- offset(type, rest) do
          offset + n
        end
      _ ->
        nil
    end
  end
  def offset(%__MODULE__{vars: vars}, field) do
    case Keyword.get(vars, field) do
      {offset, _} -> offset
      _           -> nil
    end
  end
  def offset(shape, [field | rest]) when is_map(shape) do
    result = shape |> Enum.reduce({:not_found, 0}, fn
      {^field, _}, {:not_found, offset} -> {:ok, offset}
      {_, type}, {:not_found, offset} -> {:not_found, offset + offset(type, rest)}
      _, result -> result
    end)
    with {:ok, offset} <- result do
      offset
    else
      _ -> nil
    end
  end
  def offset(shape, []), do: size(shape)
  def offset(shape, field), do: offset(shape, [field])

  def inspect_structure(%__MODULE__{} = mem, opts \\ []) do
    vars = mem.vars |> Enum.map(fn {k, v} -> {Cuda.Graph.Node.string_id(k), v} end)
    w1 = vars |> Enum.map(& elem(&1, 0)) |> Enum.map(&String.length/1) |> Enum.max()
    w2 = vars |> Enum.map(& elem(elem(&1, 1), 0)) |> Enum.map(&String.length("#{&1}")) |> Enum.max()
    vars = vars
           |> Enum.map(fn {k, {o, t}} ->
             name = String.pad_trailing(k, w1, " ")
             offset = String.pad_leading("#{o}", w2, " ")
             "#{name} | #{offset} | #{inspect t}"
           end)
           |> Enum.join("\n")
    label = case Keyword.get(opts, :label) do
      nil   -> ""
      label -> " #{inspect(label)}"
    end
    IO.puts("\nMemory#{label}. Type: #{mem.type}\n#{vars}")
    mem
  end

  def arity({_, arity}) do
    size(arity)
  end
  def arity(_), do: 1

  def type({t, _}), do: t
  def type(t), do: t

  def pack(:zero, t) when is_atom(t) or is_bitstring(t), do: pack(0, t)
  def pack(_, {:skip, bytes}) when bytes < 0, do: <<>>
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
  def pack(x, :f16), do: Float16.pack(x)
  def pack(x, :f32), do: <<x::float-little-size(32)>>
  def pack(x, :f64), do: <<x::float-little-size(64)>>
  def pack(x, "f16"), do: pack(x, :f16)
  def pack(x, "f32"), do: pack(x, :f32)
  def pack(x, "f64"), do: pack(x, :f64)
  def pack(x, {type, arity}) when not is_tuple(arity), do: pack(x, {type, {arity}})
  def pack(x, {type, arity}) when is_list(x) do
    arity = size(arity)
    x = List.flatten(x)
    if length(x) == arity do
      x |> Enum.map(& pack(&1, type)) |> Enum.join
    else
      raise RuntimeError, message: "Arity of array #{inspect x} should be #{arity}"
    end
  end
  def pack(:zero, {type, arity}) do
    size = size(arity) * size(type)
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
  def pack(x, %__MODULE__{vars: vars}) when is_map(x) do
    vars
    |> Enum.map(fn {k, {_offset, type}} ->
      #IO.inspect({k, type, Map.get(x, k, :zero)})
      pack(Map.get(x, k, :zero), type)
    end)
    |> Enum.join()
  end
  def pack(%{} = x, %Shape{type: type, skip: %{} = skip}) do
    x
    |> Map.to_list()
    |> Enum.reduce("", fn {key, val}, acc ->
      t = Map.get(type, key)
      s = Map.get(skip, key)
      acc <> pack(val, %Shape{type: t, skip: s})
    end)
  end
  def pack(x, %Shape{type: type, skip: {sbefore, safter}}) do
    pack(0, {:skip, sbefore}) <> pack(x, type) <> pack(0, {:skip, safter})
  end
  def pack(x, %Shape{type: type, skip: skip}) when is_integer(skip) do
    pack(x, type) <> pack(0, {:skip, skip})
  end
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
  def unpack(x, :f16), do: Float16.unpack(x)
  def unpack(<<x::float-little-size(32)>>, :f32), do: x
  def unpack(<<x::float-little-size(64)>>, :f64), do: x
  def unpack(x, "f16"), do: unpack(x, :f16)
  def unpack(x, "f32"), do: unpack(x, :f32)
  def unpack(x, "f64"), do: unpack(x, :f64)
  def unpack(x, {type, arity}) when is_tuple(arity) do
    arity = arity |> Tuple.to_list |> Enum.reverse
    {list, _} = unpack_list(x, {type, arity})
    list
    |> unp_flat()
  end
  def unpack(x, {type, arity}) when not is_tuple(arity) do
    {list, _} = unpack_list(x, {type, [arity]})
    list
    |> unp_flat()
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
    |> unp_flat()
  end
  def unpack(x, %__MODULE__{vars: vars}) do
    vars
    |> Enum.map(fn {k, {offset, type}} ->
      x = x
          |> binary_part(offset, size(type))
          |> unpack(type)
      {k, x}
    end)
    |> Enum.into(%{})
  end
  def unpack(x, %Shape{type: %{} = type, skip: %{} = skip}) do
    type = type
    |> Map.to_list()
    |> Enum.map(fn {key, val} ->
      val = is_list(val) && val || [val]
      case Map.get(skip, key, 0)  do
        {sbefore, safter} -> {key, [{:skip, sbefore} | val] ++ [{:skip, safter}]}
        0                 -> {key, val}
        s                 -> {key, val ++ [{:skip, s}]}
      end
    end)
    |> Enum.into(%{})
    x
    |> unpack(type)
  end
  def unpack(x, %Shape{type: type, skip: {sbefore, safter}}) do
    size = byte_size(x) - sbefore - safter
    x
    |> binary_part(sbefore, size)
    |> unpack(type)
  end
  def unpack(x, %Shape{type: type, skip: skip}) when is_integer(skip) and skip < 0 do
    unpack(x, type)
  end
  def unpack(x, %Shape{type: type, skip: skip}) when is_integer(skip) do
    size = byte_size(x) - skip
    <<data::binary-size(size), _::binary>> = x
    unpack(data, type)
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
    |> unp_flat()
  end
  def unpack(_, _), do: nil

  defp unpack_list(x, {type, [arity]}) do
    size = size(type)
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
  defp unpack_list(x, {:skip, bytes}) when bytes < 0 do
    {[], x}
  end
  defp unpack_list(x, {:skip, bytes}) do
    <<_::binary-size(bytes), rest::binary>> = x
    {[], rest}
  end
  defp unpack_list(x, type) do
    size = size(type)
    #IO.inspect({x, type, size, byte_size(x)})
    <<x::binary-size(size), rest::binary>> = x
    {[unpack(x, type)], rest}
  end

  defp unp_flat(%{} = value) do
    value
    |> Map.to_list()
    |> Enum.map(fn {k, v} -> {k, unp_flat(v)} end)
    |> Enum.into(%{})
  end
  defp unp_flat([value]) when is_list(value), do: unp_flat(value)
  defp unp_flat(value),                       do: value

  @type_re ~r/(\d+)/
  def size(:i8),  do: 1
  def size(:i16), do: 2
  def size(:i32), do: 4
  def size(:i64), do: 8
  def size(:u8),  do: 1
  def size(:u16), do: 2
  def size(:u32), do: 4
  def size(:u64), do: 8
  def size(:f16), do: 2
  def size(:f32), do: 4
  def size(:f64), do: 8
  def size({:skip, n}) when n < 0, do: 0
  def size({:skip, n}), do: n
  def size(type) when is_atom(type) or is_bitstring(type) do
    case Regex.run(@type_re, "#{type}", capture: :all_but_first) do
      [n] -> div(String.to_integer(n), 8)
      _   -> 0
    end
  end
  def size(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list
    |> Enum.map(&size/1)
    |> Enum.reduce(1, &Kernel.*/2)
  end
  def size(i) when is_integer(i), do: i
  def size(l) when is_list(l) do
    l |> Enum.map(&size/1) |> Enum.reduce(0, &+/2)
  end
  def size(%__MODULE__{vars: vars}) do
    vars |> Enum.reduce(0, fn {_, {_, t}}, a -> a + size(t) end)
  end
  def size(%Shape{type: nil}), do: 0
  def size(%Shape{type: type, skip: %{} = skip}) do
    ssize = skip
    |> Map.to_list()
    |> Enum.reduce(0, fn
      {_, {v1, v2}}, acc -> acc + v1 + v2
      {_, v}, acc        -> acc + v
    end)
    size(type) + ssize
  end
  def size(%Shape{type: type, skip: {sbefore, safter}}) do
    size({:skip, sbefore + safter}) + size(type)
  end
  def size(%Shape{type: type, skip: skip}) when is_integer(skip) do
    size({:skip, skip}) + size(type)
  end
  def size(m) when is_map(m) do
    m
    |> Enum.map(fn {_, v} -> size(v) end)
    |> Enum.reduce(0, &+/2)
  end
  def size(_), do: 0

  def size_equal?(x, y) do
    size(x) == size(y)
  end
end
