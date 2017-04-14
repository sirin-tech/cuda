defmodule TemplateTest do
  use ExUnit.Case
  import Cuda.Template

  describe "ptx_eval/2" do
    test "String.upcase/1 calling with parameter stored in context variable" do
      template = ~s[<%= upcase(var(ctx, :text)) %>]
      ctx = %Cuda.Template.Context{env: elem(Cuda.Env.create(), 1), var: %{text: "Hello, EEx!"}}
      assert "HELLO, EEX!" == ptx_eval(template, [context: ctx, ptx_helpers: [String]])
    end

    test "Add 10 to environment variable int_size" do
      template = ~s[<%= env(ctx, :int_size) + var(ctx, :number) %>]
      ctx = %Cuda.Template.Context{env: %Cuda.Env{int_size: 10}, var: %{number: 10}}
      assert "20" == ptx_eval(template, [context: ctx, ptx_helpers: [Kernel]])
    end
  end

  describe "c_eval/2" do
    test "String.upcase/1 calling with parameter stored in context variable" do
      template = ~s[<%= upcase(var(ctx, :text)) %>]
      ctx = %Cuda.Template.Context{env: elem(Cuda.Env.create(), 1), var: %{text: "Hello, EEx!"}}
      assert "HELLO, EEX!" == c_eval(template, [context: ctx, c_helpers: [String]])
    end

    test "Add 10 to environment variable int_size" do
      template = ~s[<%= env(ctx, :int_size) + var(ctx, :number) %>]
      ctx = %Cuda.Template.Context{env: %Cuda.Env{int_size: 10}, var: %{number: 10}}
      assert "20" == c_eval(template, [context: ctx, c_helpers: [Kernel]])
    end
  end
end
