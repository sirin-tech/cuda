defmodule Cuda.MemoryTest do
  use ExUnit.Case
  alias Cuda.Memory.Shape
  import Cuda.Memory

  describe "pack/2" do
    test "pack a 1d array [1, 2, 3, 4] of i8" do
      p = pack([1, 2, 3, 4], {:i8, 4})
      assert p == <<1, 2, 3, 4>>
    end

    test "pack a 2d array [[1, 2, 3, 4], [5, 6, 7, 8]] of i8" do
      p = pack([[1, 2, 3, 4], [5, 6, 7, 8]], {:i8, {4, 2}})
      assert p == <<1, 2, 3, 4, 5, 6, 7, 8>>
    end

    test "pack a 2d array [[1, 2, 3, 4], [5, 6, 7, 8]] of i8 with %Shape{} usage " do
      p = pack([[1, 2, 3, 4], [5, 6, 7, 8]], %Shape{type: {:i8, {4, 2}}})
      assert p == <<1, 2, 3, 4, 5, 6, 7, 8>>
    end

    test "pack a 2d array [[1, 2, 3, 4], [5, 6, 7, 8]] of i8 with %Shape{} usage and skip 4 bytes after" do
      p = pack([[1, 2, 3, 4], [5, 6, 7, 8]], %Shape{type: {:i8, {4, 2}}, skip: 4})
      assert p == <<1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0>>
    end

    test "pack a 2d array [[1, 2, 3, 4], [5, 6, 7, 8]] of i8 with %Shape{} usage, skip 2 bytes before, and 4 bytes after" do
      p = pack([[1, 2, 3, 4], [5, 6, 7, 8]], %Shape{type: {:i8, {4, 2}}, skip: {2, 4}})
      assert p == <<0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0>>
    end

    test "pack a 2d array [[1, 2, 3, 4], [1.2, 13.3, 5.6, 12.1]] of i8, f32" do
      p = pack([[1, 2, 3, 4], [1.2, 13.3, 5.6, 12.1]], [{:i8, 4}, {:f32, 4}])
      assert p == <<1, 2, 3, 4, 154, 153, 153, 63, 205, 204, 84, 65, 51, 51, 179, 64, 154, 153, 65, 65>>
    end

    test "pack a map %{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]} of i8, f32" do
      p = pack(%{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]}, %{a: {:i8, 4}, b: {:f32, 4}})
      assert p == <<1, 2, 3, 4, 154, 153, 153, 63, 205, 204, 84, 65, 51, 51, 179, 64, 154, 153, 65, 65>>
    end

    test "pack a map %{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]} of i8, f32, with %Shape{} usage" do
      p = pack(%{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]}, %Shape{type: %{a: {:i8, 4}, b: {:f32, 4}}})
      assert p == <<1, 2, 3, 4, 154, 153, 153, 63, 205, 204, 84, 65, 51, 51, 179, 64, 154, 153, 65, 65>>
    end

    test "pack a map %{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]} of i8, f32, with %Shape{} usage, skip 4 bytes before for 'a' and 5 bytes before, and 2 bytes after for 'k'" do
      p = pack(%{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]}, %Shape{type: %{a: {:i8, 4}, b: {:f32, 4}}, skip: %{a: {4, 0}, b: {5, 2}}})
      assert p == <<0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 154, 153, 153, 63, 205, 204, 84, 65, 51, 51, 179, 64, 154, 153, 65, 65, 0, 0>>
    end

    test "pack a 3d array [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] of i8" do
      p = pack([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], {:i8, {4, 3}})
      assert p == <<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12>>
    end

    test "pack a f32 value and i8 1d array [65.7, [1, 2, 3, 4]]" do
      p = pack([65.7, [1, 2, 3, 4]], [:f32, {:i8, 4}])
      assert p == <<102, 102, 131, 66, 1, 2, 3, 4>>
    end

    test "pack a zero f32 value" do
      p = pack(:zero, :f32)
      assert p == <<0, 0, 0, 0>>
    end
  end

  describe "unpack/2" do
    test "unpack a 1d array [1, 2, 3, 4] of i8" do
      p = unpack(<<1, 2, 3, 4>>, {:i8, 4})
      assert p == [1, 2, 3, 4]
    end

    test "unpack a 2d array [[1, 2, 3, 4], [5, 6, 7, 8]] of i8" do
      p = unpack(<<1, 2, 3, 4, 5, 6, 7, 8>>, {:i8, {4, 2}})
      assert p == [[1, 2, 3, 4], [5, 6, 7, 8]]
    end

    test "unpack a 2d array [[1, 2, 3, 4], [5, 6, 7, 8]] of i8 with %Shape{} usage " do
      p = unpack(<<1, 2, 3, 4, 5, 6, 7, 8>>, %Shape{type: {:i8, {4, 2}}})
      assert p == [[1, 2, 3, 4], [5, 6, 7, 8]]
    end

    test "unpack a 2d array [[1, 2, 3, 4], [5, 6, 7, 8]] of i8 with %Shape{} usage and skip 4 bytes after" do
      p = unpack(<<1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0>>, %Shape{type: {:i8, {4, 2}}, skip: 4})
      assert p == [[1, 2, 3, 4], [5, 6, 7, 8]]
    end

    test "unpack a 2d array [[1, 2, 3, 4], [5, 6, 7, 8]] of i8 with %Shape{} usage, skip 2 bytes before, and 4 bytes after" do
      p = unpack(<<0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0>>, %Shape{type: {:i8, {4, 2}}, skip: {2, 4}})
      assert p == [[1, 2, 3, 4], [5, 6, 7, 8]]
    end

    test "unpack a 2d array [[1, 2, 3, 4], [1.2, 13.3, 5.6, 12.1]] of i8, f32" do
      p = <<1, 2, 3, 4, 154, 153, 153, 63, 205, 204, 84, 65, 51, 51, 179, 64, 154, 153, 65, 65>>
      |> unpack([{:i8, 4}, {:f32, 4}])
      |> rnd()
      assert p == [[1, 2, 3, 4], [1.2, 13.3, 5.6, 12.1]]
    end

    test "unpack a map %{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]} of i8, f32" do
      p = <<1, 2, 3, 4, 154, 153, 153, 63, 205, 204, 84, 65, 51, 51, 179, 64, 154, 153, 65, 65>>
      |> unpack(%{a: {:i8, 4}, b: {:f32, 4}})
      |> rnd()
      assert p == %{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]}
    end

    test "unpack a map %{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]} of i8, f32, with %Shape{} usage" do
      p = <<1, 2, 3, 4, 154, 153, 153, 63, 205, 204, 84, 65, 51, 51, 179, 64, 154, 153, 65, 65>>
      |> unpack(%Shape{type: %{a: {:i8, 4}, b: {:f32, 4}}})
      |> rnd()
      assert p == %{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]}
    end

    test "unpack a map %{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]} of i8, f32, with %Shape{} usage, skip 4 bytes before for 'a' and 5 bytes before, and 2 bytes after for 'k'" do
      p = <<0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 154, 153, 153, 63, 205, 204, 84, 65, 51, 51, 179, 64, 154, 153, 65, 65, 0, 0>>
      |> unpack(%Shape{type: %{a: {:i8, 4}, b: {:f32, 4}}, skip: %{a: {4, 0}, b: {5, 2}}})
      |> rnd()
      assert p == %{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.1]}
    end

    test "unpack a 3d array [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] of i8" do
      p = unpack(<<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12>>, {:i8, {4, 3}})
      assert p == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    end

    test "pack a f32 value and i8 1d array [65.7, [1, 2, 3, 4]]" do
      p = <<102, 102, 131, 66, 1, 2, 3, 4>>
      |> unpack([:f32, {:i8, 4}])
      |> rnd()
      assert p == [65.7, [1, 2, 3, 4]]
    end
  end

  describe "size/1" do
    test "Size of single type" do
      assert size(:f32) == 4
    end

    test "Size of type's list" do
      assert size([:f32, {:i8, 3}]) == 7
    end

    test "Size of type's tuple" do
      assert size({:i8, {3, 3, 2}}) == 18
    end

    test "Size of type's map" do
      assert size(%{a: {:i8, {3, 3, 2}}, b: :f32}) == 22
    end

    test "Size of type's Shape" do
      assert size(%Shape{type: %{a: {:i8, {3, 3, 2}}, b: :f32}}) == 22
    end

    test "Size of type's Shape with skip" do
      assert size(%Shape{type: %{a: {:i8, {3, 3, 2}}, b: :f32}, skip: %{a: 2, b: {3, 2}}}) == 29
      assert size(%Shape{type: %{a: {:i8, {3, 3, 2}}, b: :f32}, skip: %{b: {3, 2}}})       == 27
      assert size(%Shape{type: %{a: {:i8, {3, 3, 2}}, b: :f32}, skip: 2})                  == 24
      assert size(%Shape{type: %{a: {:i8, {3, 3, 2}}, b: :f32}, skip: {3, 2}})             == 27
    end
  end

  describe "size_equal?/2" do
    test "general" do
      assert size_equal?(:f32, :f32)
    end

    test "different type sizes" do
      refute size_equal?(:f32, :i8)
    end

    test "equal types sizes" do
      assert size_equal?(:f32, {:i8, 4})
    end
  end

  def rnd(x, precision \\ 1)
  def rnd(%{} = x, precision) do
    x
    |> Map.to_list()
    |> Enum.map(fn {key, val} -> {key, rnd(val, precision)} end)
    |> Enum.into(%{})
  end
  def rnd(x, precision) when is_list(x) do
    x
    |> Enum.map(&rnd(&1, precision))
  end
  def rnd(x, precision) when is_float(x) do
    Float.round(x, precision)
  end
  def rnd(x, _), do: x
end
