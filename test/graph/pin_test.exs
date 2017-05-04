defmodule Cuda.Graph.PinTest do
  use ExUnit.Case
  alias Cuda.Graph.Pin
  import Pin

  describe "data_type/1" do
    test "accepts simple types" do
      assert data_size(%Pin{data_type: :i8}) == 1
      assert data_size(%Pin{data_type: :i16}) == 2
      assert data_size(%Pin{data_type: :i32}) == 4
      assert data_size(%Pin{data_type: :i64}) == 8
      assert data_size(%Pin{data_type: :u8}) == 1
      assert data_size(%Pin{data_type: :u16}) == 2
      assert data_size(%Pin{data_type: :u32}) == 4
      assert data_size(%Pin{data_type: :u64}) == 8
      assert data_size(%Pin{data_type: :f16}) == 2
      assert data_size(%Pin{data_type: :f32}) == 4
      assert data_size(%Pin{data_type: :f64}) == 8
    end

    test "accepts unknown types with size qulifier" do
      assert data_size(%Pin{data_type: :test128}) == 16
    end

    test "accepts tuples" do
      assert data_size(%Pin{data_type: {:i16, :i16}}) == 4
      assert data_size(%Pin{data_type: {:f16, :i32, :i32}}) == 32
      assert data_size(%Pin{data_type: {:i16, {:i32, :i32}, :i16}}) == 64
    end
  end
end
