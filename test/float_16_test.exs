defmodule Cuda.Float16Test do
  use ExUnit.Case
  import Cuda.Float16

  describe "pack/1" do
    test "Positive value without fraction" do
      assert pack(2) == <<0::size(1),16::size(5),0::size(10)>>
    end

    test "Negative value without fraction" do
      assert pack(-2) == <<1::size(1),16::size(5),0::size(10)>>
    end

    test "Positive float value" do
      assert pack(155.625) == <<0::size(1),22::size(5),221::size(10)>>
    end

    test "Negative float value" do
      assert pack(-155.625) == <<1::size(1),22::size(5),221::size(10)>>
    end

    test "Maximal normal number" do
      assert pack(65504) == <<0::size(1),30::size(5),1023::size(10)>>
    end

    test "Minimal normal number" do
      assert pack(0.000061) == <<0::size(1),0::size(5),1022::size(10)>>
    end

    test "Minimal subnormal number" do
      assert pack(0.00000006) == <<0::size(1),0::size(5),1::size(10)>>
    end

    test "Zero" do
      assert pack(0) == <<0::size(1),0::size(5),0::size(10)>>
    end
  end

  describe "unpack/1" do
    test "Positive value without fraction" do
      assert unpack(<<0::size(1),16::size(5),0::size(10)>>) == 2.0
    end

    test "Negative value without fraction" do
      assert unpack(<<1::size(1),16::size(5),0::size(10)>>) == -2.0
    end

    test "Positive float value" do
      assert unpack(<<0::size(1),22::size(5),221::size(10)>>) == 155.625
    end

    test "Negative float value" do
      assert unpack(<<1::size(1),22::size(5),221::size(10)>>) == -155.625
    end

    test "Maximal normal number" do
      assert unpack(<<0::size(1),30::size(5),1023::size(10)>>) == 65504
    end

    test "Minimal normal number" do
      assert <<0::size(1),0::size(5),1022::size(10)>>
      |> unpack()
      |> Float.round(6) == 0.000061
    end

    test "Minimal subnormal number" do
      assert <<0::size(1),0::size(5),1::size(10)>>
      |> unpack()
      |> Float.round(9) == 0.00000006
    end

    test "Zero" do
      assert unpack(<<0::size(1),0::size(5),0::size(10)>>) == 0
    end

    test "Not a number" do
      assert unpack(<<0::size(1),31::size(5),1::size(10)>>) == :not_a_number
    end

    test "Positive infinity" do
      assert unpack(<<0::size(1),31::size(5),0::size(10)>>) == :positive_infinity
    end

    test "Negative infinity" do
      assert unpack(<<1::size(1),31::size(5),0::size(10)>>) == :negative_infinity
    end
  end
end
