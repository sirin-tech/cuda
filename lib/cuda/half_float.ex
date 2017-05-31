defmodule Cuda.HalfFloat do
  # sign     - 1  bit
  # exponent - 5  bits
  # mantiss  - 10 bits
  #
  # Exponent bias = 15 = 01111

  def max_power(value, is_power \\ false) do
    b = Integer.digits(value, 2)
    p = Enum.reduce_while(b, length(b) - 1, fn
      _, 0 -> {:halt, 0}
      1, p -> {:halt, p}
      0, p -> {:cont, p - 1}
    end)
    if is_power,
      do: p,
    else: round(:math.pow(2, p))
  end

  def float_binary(value, precision \\ 9) when is_float(value)  do
    i = Float.floor(value)
    fract = value - i
    i = Integer.digits(round(i), 2)
    f = fb_fract_binary(fract, precision)
    {i, f}
  end

  def normalize({[0], [0]}), do: {0, 0}
  def normalize({[0], fract}) do
    case Enum.find_index(fract, &(&1 == 1)) do
      nil -> {0, 0}
      ind ->
        mantiss = Enum.drop(fract, ind)
        exp = Integer.digits(-ind + 15, 2)
        {mantiss, exp}
    end
  end

  defp fb_fract_binary(0.0, _, result), do: result
  defp fb_fract_binary(_, 0, result),   do: result
  defp fb_fract_binary(fract, precision, result \\ []) do
    fract = fract * 2
    if fract >= 1 do
      fb_fract_binary(fract - 1, precision - 1, result ++ [1])
    else
      fb_fract_binary(fract, precision - 1, result ++ [0])
    end
  end
end
