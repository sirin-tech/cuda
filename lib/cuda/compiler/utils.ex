defmodule Cuda.Compiler.Utils do
  alias Cuda.Graph.{Node, NodeProto, Pin}
  alias Cuda.Memory
  import Node, only: [input_pin_types: 0, output_pin_types: 0]

  # TODO: Fix negative skips.
  #       reproduced by `mix test test/network_test.exs:85`,
  #       negative skip in input of {:back_propagation, :fc, :fc_node}
  def put_pins_shapes(%{assigns: %{pin_offsets: offsets}} = node) do
    #IO.inspect({node.id, offsets})
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
      %Memory{vars: inputs.vars ++ outputs.vars}
    end# |> Memory.inspect_structure(label: node.id)
    memory = node.assigns
             |> Map.get(:memory, %{})
             |> Map.put(:pins, pins)
    NodeProto.assign(node, memory: memory)
  end
  def put_pins_shapes(%{pins: _} = node) do
    pins   = node.pins |> pins_shape()# |> Memory.inspect_structure(label: node.id)
    memory = node.assigns
             |> Map.get(:memory, %{})
             |> Map.put(:pins, pins)
    NodeProto.assign(node, :memory, memory)
  end
  def put_pins_shapes(node), do: node

  def pins_shape(pins) do
    {vars, _} = pins |> Enum.map_reduce(0, fn pin, offset ->
      size = Pin.data_size(pin)
      {{pin.id, {offset, pin.data_type}}, offset + size}
    end)
    %Memory{vars: vars}
  end
  def pins_shape(pins, offsets, _pin_size) do
    pin_ids = pins |> Enum.map(& &1.id)
    offsets = offsets |> Enum.sort_by(fn {_, {o, _}} -> o end)
    pin_size = offsets
               |> Enum.map(& elem(elem(&1, 1), 1))
               |> Enum.reduce(0, &+/2)
    [{_, {first_offset, _}} | _] = offsets
    pin_size = pin_size + first_offset
    {k, o} = offsets
             |> Enum.filter(fn {k, _} -> k in pin_ids end)
             |> Enum.unzip()
    s = o |> Enum.map(& elem(&1, 0))
    s = Enum.chunk(s ++ [pin_size], 2, 1) |> Enum.map(fn [a, b] -> b - a end)
    vars = [k, o, s]
           |> Enum.zip()
           |> Enum.with_index()
           |> Enum.map(fn {{pin_id, {offset, data_size}, size}, idx} ->
             pin = Enum.find(pins, & &1.id == pin_id)
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
