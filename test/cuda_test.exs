defmodule Cuda.CudaTest do
  use ExUnit.Case
  doctest Cuda

  describe "stream/3" do
    # TODO: Sometimes this test fails with following message:
    #   1) test stream/3 process batch calculations (Cuda.CudaTest)
    #      test/cuda_test.exs:6
    #      Assertion with == failed
    #      code:  result == <<3::little-32, 4::little-32, 5::little-32, 6::little-32, 7::little-32, 8::little-32, 9::little-32, 10::little-32>>
    #      left:  <<1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0,
    #               0, 0, 7, 0, 0, 0, 8, 0, 0, 0>>
    #      right: <<3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0,
    #               0, 0, 9, 0, 0, 0, 10, 0, 0, 0>>
    #      stacktrace:
    #        test/cuda_test.exs:67: (test)
    #
    # We should reproduce this error and rewrite test

    test "process batch calculations" do
      # x[i] = x[i] + 2
      ptx1 = """
             .version 4.3
             .target sm_20
             .address_size 64
             .visible .entry ptx1(
               .param .u64 .ptr .global input,
             	 .param .u64 .ptr .global output
             ) {
               .reg .u64 %p<3>;
               .reg .u32 %r<2>;
               ld.param.u64 %p0, [input];
             	 ld.param.u64 %p1, [output];
	             mov.u32 %r1, %tid.x;
               mul.wide.u32 %p2, %r1, 4;
               add.u64 %p0, %p0, %p2;
               add.u64 %p1, %p1, %p2;
               ld.global.u32 %r0, [%p0];
             	 add.u32 %r0, %r0, 2;
               st.global.u32 [%p1], %r0;
             }
             """

      # x[i] = x[i] - 1
      ptx2 = """
             .version 4.3
             .target sm_20
             .address_size 64
             .visible .entry ptx2(
               .param .u64 .ptr .global input,
             	 .param .u64 .ptr .global output
             ) {
               .reg .u64 %p<3>;
               .reg .u32 %r<2>;
               ld.param.u64 %p0, [input];
             	 ld.param.u64 %p1, [output];
	             mov.u32 %r1, %tid.x;
               mul.wide.u32 %p2, %r1, 4;
               add.u64 %p0, %p0, %p2;
               add.u64 %p1, %p1, %p2;
               ld.global.u32 %r0, [%p0];
             	 sub.u32 %r0, %r0, 1;
               st.global.u32 [%p1], %r0;
             }
             """

      data = <<1::little-32, 2::little-32, 3::little-32, 4::little-32,
               5::little-32, 6::little-32, 7::little-32, 8::little-32>>
      {:ok, cuda}   = Cuda.start_link()
      {:ok, input}  = Cuda.memory_load(cuda, data)
      {:ok, output} = Cuda.memory_load(cuda, <<0::size(8)-unit(32)>>)
      {:ok, module} = Cuda.compile(cuda, [ptx1, ptx2])

      # x[i] = x[i] + 2 - 1 + 2 - 1
      batch = [[{:run, {"ptx1", {8, 1, 1}, {1, 1, 1}, [input, output]}},
                {:run, {"ptx2", {8, 1, 1}, {1, 1, 1}, [output, input]}},
                {:run, {"ptx1", {8, 1, 1}, {1, 1, 1}, [input, output]}},
                {:run, {"ptx2", {8, 1, 1}, {1, 1, 1}, [output, input]}}]]
      :ok = Cuda.stream(cuda, module, batch)
      {:ok, result} = Cuda.memory_read(cuda, input)
      assert result == <<3::little-32, 4::little-32, 5::little-32, 6::little-32,
                         7::little-32, 8::little-32, 9::little-32, 10::little-32>>
    end
  end
end
