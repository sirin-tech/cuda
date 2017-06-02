defmodule Cuda.TemplateTest do
  use ExUnit.Case
  import Cuda.Test.CudaHelpers

  alias Cuda.Compiler.Context
  import Cuda.Template

  describe "ptx_eval/2" do
    test "String.upcase/1 calling with parameter stored in context variable" do
      template = ~s[<%= upcase(var(:text)) %>]
      ctx = %Context{env: env(), assigns: %{vars: %{text: "Hello, EEx!"}}}
      assert "HELLO, EEX!" == ptx_eval(template, [context: ctx, ptx_helpers: [String]])
    end

    test "Add 10 to environment variable int_size" do
      template = ~s[<%= env(:int_size) + var(:number) %>]
      ctx = %Context{env: env(int_size: 10), assigns: %{vars: %{number: 10}}}
      assert "20" == ptx_eval(template, [context: ctx])
    end
  end

  describe "c_eval/2" do
    test "String.upcase/1 calling with parameter stored in context variable" do
      template = ~s[<%= upcase(var(ctx, :text)) %>]
      ctx = %Context{env: env(), assigns: %{vars: %{text: "Hello, EEx!"}}}
      assert "HELLO, EEX!" == c_eval(template, [context: ctx, c_helpers: [String]])
    end

    test "Add 10 to environment variable int_size" do
      template = ~s[<%= env(ctx, :int_size) + var(ctx, :number) %>]
      ctx = %Context{env: env(int_size: 10), assigns: %{vars: %{number: 10}}}
      assert "20" == c_eval(template, [context: ctx])
    end
  end
end
