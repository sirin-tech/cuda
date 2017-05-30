defmodule Cuda.Compiler.Utils do
  alias Cuda.Graph.{Node, NodeProto, Pin}
  alias Cuda.Memory
  import Node, only: [input_pin_types: 0, output_pin_types: 0]

  def put_pins_shapes(%{assigns: %{pin_offsets: offsets}} = node) do
    size = case Map.get(node.assigns, :pin_size) do
      nil  -> node.pins |> Enum.map(&Pin.data_size/1) |> Enum.reduce(0, &+/2)
      size -> size
    end
    o = offsets |> Map.values()
    pins = if o == Enum.uniq(o) do
      node.pins |> pins_shape(offsets, size)
    else
      inputs  = node
                |> NodeProto.pins(input_pin_types())
                |> pins_shape(offsets, size)
      outputs = node
                |> NodeProto.pins(output_pin_types())
                |> pins_shape(offsets, size)
      Memory.merge(inputs, outputs)
    end
    memory = node.assigns
             |> Map.get(:memory, %{})
             |> Map.put(:pins, pins)
    NodeProto.assign(node, memory: memory)
  end
  def put_pins_shapes(%{pins: _} = node) do
    pins   = node.pins |> pins_shape()
    memory = node.assigns
             |> Map.get(:memory, %{})
             |> Map.put(:pins, pins)
    NodeProto.assign(node, :memory, memory)
  end
  def put_pins_shapes(node), do: node

  def pins_shape(pins) do
    {vars, _} = pins |> Enum.reduce({[], 0}, fn pin, {vars, offset} ->
      size = Pin.data_size(pin)
      {vars ++ [{pin.id, {offset, pin.type}}], offset + size}
    end)
    %Memory{vars: vars}
  end
  def pins_shape(pins, offsets, pin_size) do
    pin_ids = pins |> Enum.map(& &1.id)
    {k, o} = offsets
             |> Enum.filter(fn {k, _} -> k in pin_ids end)
             |> Enum.sort_by(fn {_, o} -> o end)
             |> Enum.unzip()
    s = Enum.chunk(o ++ [pin_size], 2, 1) |> Enum.map(fn [a, b] -> b - a end)
    vars = [k, o, s]
           |> Enum.zip()
           |> Enum.with_index()
           |> Enum.map(fn {{pin_id, offset, size}, idx} ->
             pin = Enum.find(pins, & &1.id == pin_id)
             data_size = Pin.data_size(pin)
             skip_before = if idx == 0 and offset > 0, do: offset, else: 0
             skip_after = case size - data_size do
               skip when skip < 0 -> pin_size + skip
               skip               -> skip
             end
             shape = case {skip_before, skip_after} do
               {0, 0} -> pin.data_type
               {0, a} -> %Memory.Shape{skip: a, type: pin.data_type}
               {b, a} -> %Memory.Shape{skip: {b, a}, type: pin.data_type}
             end
             {pin_id, {offset, shape}}
           end)
    %Memory{vars: vars}
  end

  def wrap_pins(pins) when is_map(pins) do
    pins |> Enum.map(fn {k, v} -> {k, [v]} end) |> Enum.into(%{})
  end
  def wrap_pins(pins), do: pins

  def unwrap_pins(pins) when is_map(pins) do
    pins |> Enum.map(fn {k, [v | _]} -> {k, v} end) |> Enum.into(%{})
  end
  def unwrap_pins(pins), do: pins
end
