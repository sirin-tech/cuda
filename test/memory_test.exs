defmodule Cuda.MemoryTest do
  use ExUnit.Case
  import Cuda.Memory

  describe "pack/2" do
    test "pack a 1d array [1, 2, 3, 4] of i8" do
      p = pack([1, 2, 3, 4], {:i8, 4})
      assert p == <<1, 2, 3, 4>>
    end

    test "pack a 2d array [[1, 2, 3, 4], [1, 2, 3]] of i8" do
      p = pack([[1, 2, 3, 4], [1, 2, 3]], {:i8, [4, 3]})
      assert p == <<1, 2, 3, 4, 1, 2, 3>>
    end

    test "pack a 2d array [[1, 2, 3, 4], [1.2, 13.3, 5.6, 12.129]] of i8, f32" do
      p = pack([[1, 2, 3, 4], [1.2, 13.3, 5.6, 12.129]], [{:i8, 4}, {:f32, 4}])
      assert p == <<1, 2, 3, 4, 154, 153, 153, 63, 205, 204, 84, 65, 51, 51, 179, 64, 98, 16, 66, 65>>
    end

    test "pack a map %{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.129]} of i8, f32" do
      p = pack(%{a: [1, 2, 3, 4], b: [1.2, 13.3, 5.6, 12.129]}, %{a: {:i8, 4}, b: {:f32, 4}})
      assert p == <<1, 2, 3, 4, 154, 153, 153, 63, 205, 204, 84, 65, 51, 51, 179, 64, 98, 16, 66, 65>>
    end

    test "pack a 3d array [[1, 2, 3, 4], [1, 2, 3], [1, 2]] of i8" do
      p = pack([[1, 2, 3, 4], [1, 2, 3], [1, 2]], {:i8, [4, 3, 2]})
      assert p == <<1, 2, 3, 4, 1, 2, 3, 1, 2>>
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
end
