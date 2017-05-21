defmodule Cuda.Compiler.Utils do
  alias Cuda.Graph.{Node, NodeProto, Pin}
  import Node, only: [input_pin_types: 0, output_pin_types: 0]

  def put_pins_shapes(%{assigns: %{pin_offsets: offsets}} = node) do
    size = case Map.get(node.assigns, :pin_size) do
      nil  -> node.pins |> Enum.map(&Pin.data_size/1) |> Enum.reduce(0, &+/2)
      size -> size
    end
    inputs  = node
              |> NodeProto.pins(input_pin_types())
              |> pins_shape(offsets, size)
    outputs = node
              |> NodeProto.pins(output_pin_types())
              |> pins_shape(offsets, size)
    NodeProto.assign(node, inputs_shape: inputs, outputs_shape: outputs)
  end
  def put_pins_shapes(%{pins: _} = node) do
    inputs  = input_pin_types() |> NodeProto.pins() |> pins_shape()
    outputs = output_pin_types() |> NodeProto.pins() |> pins_shape()
    NodeProto.assign(node, inputs_shape: inputs, outputs_shape: outputs)
  end
  def put_pins_shapes(node), do: node

  def pins_shape(pins) do
    pins |> Enum.map(& {&1.id, &1.type}) |> Enum.into(%{})
  end
  def pins_shape(pins, offsets, pin_size) do
    pin_ids = pins |> Enum.map(& &1.id)
    {k, o} = offsets
             |> Enum.filter(fn {k, _} -> k in pin_ids end)
             |> Enum.sort_by(fn {_, o} -> o end)
             |> Enum.unzip()
    s = Enum.chunk(o ++ [pin_size], 2, 1) |> Enum.map(fn [a, b] -> b - a end)
    [k, o, s]
    |> Enum.zip()
    |> Enum.with_index()
    |> Enum.map(fn {{pin_id, offset, size}, idx} ->
      pin = Enum.find(pins, & &1.id == pin_id)
      data_size = Pin.data_size(pin)
      head = if idx == 0 and offset > 0, do: [{:skip, offset}], else: []
      case size - data_size do
        0    -> {pin_id, head ++ [pin.data_type]}
        skip -> {pin_id, head ++ [pin.data_type, {:skip, skip}]}
      end
    end)
    |> Enum.into(%{})
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
