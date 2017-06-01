defmodule Cuda.HalfFloat do
  @moduledoc """
  Represents Float 16 value
  Source: http://www.softelectro.ru/ieee754.html
  """
  #======Float_16======
  # sign     - 1  bit
  # exponent - 5  bits
  # mantiss  - 10 bits
  # Exp bias - 15
  #
  #======Float_32======
  # sign     - 1  bit
  # exponent - 8  bits
  # mantiss  - 23 bits
  # Exp bias - 127
  #
  use Bitwise

  @sign_size     1
  @exponent_size 8
  @mantiss_size  23
  @exponent_bias 127

  @type float_binary :: {whole_part :: integer, fractional_part :: binary}
  @type normalize :: {mantiss :: binary, exponent :: binary}

  @doc """
  Converts number to binary representation of float 16 type
  """
  @spec pack(number) :: binary
  def pack(x) when is_integer(x), do: pack(x + 0.0)
  def pack(x) do
    f = float_binary(x)
    IO.inspect(f, label: :as_is)
    IO.inspect(f, label: :float_binary, base: :binary)
    {m, e} = normalize(f, @exponent_bias)
    IO.inspect(m, label: :mantiss, base: :binary)
    IO.inspect(e, label: :exponent, base: :binary)
    sign = x < 0 && 1 || 0
    n = size(m) - 1
    # у мантиссы убирается ведущая 1 (старший бит)
    m_cutted = m - (1 <<< n)
    m_cutted = if (@mantiss_size - n) >= 0 do
      # мантисса дополняется младшими разрядами до необходимой разрядности типа
      m_cutted <<< (@mantiss_size - n)
    else
      # если разрядность мантиссы больше установленной то, ее младшие биты
      # обрезаются  до необходимых размеров
      m_cutted >>> (n - @mantiss_size)
    end
    IO.inspect(m_cutted, label: :cutted_mantiss, base: :binary)
    <<sign::size(@sign_size), e::size(@exponent_size), m_cutted::size(@mantiss_size)>>
  end

  @doc """
  Converts binary representation of float 16 type to float value
  """
  @spec unpack(binary) :: float
  def unpack(<<sign::size(@sign_size), exp::size(@exponent_size), m::size(@mantiss_size)>>) do
    IO.inspect {:sign, sign}
    IO.inspect {:exp, exp}
    IO.inspect {:mantiss, m}
    sign = sign == 0 && 1 || -1
    # Formula explains at http://www.softelectro.ru/ieee754.html paragraph 4.2
    sign * :math.pow(2, exp - @exponent_bias)*(1 + m/(1 <<< @mantiss_size))
  end

  @doc """
  Converts float value to the binary representation
  """
  @spec float_binary(value :: float, precision :: integer) :: float_binary
  def float_binary(value, precision \\ 20) when is_float(value) do
    # get the whole part of value
    i = round(Float.floor(value))
    # get the fractional part of value
    fract = value - i
    f = fb_fract_binary(fract, precision)
    {i, f}
  end

  @spec normalize(value :: tuple, bias :: integer) :: normalize
  def normalize(value, bias \\ @exponent_bias)
  def normalize({0, {0, _}}, _), do: {0, 0}
  def normalize({0, {fract, fract_size}}, bias) do
    n = size(fract)
    exp = -(fract_size - n + 1) + bias
    {fract, exp}
  end
  def normalize({int, {fract, fract_size}}, bias) do
    int_n = size(int)
    # get exponent with bias
    exp = int_n - 1 + bias
    # целое сдвигается влево на кол-во разрядов дробной части,
    # добавляется дробная часть
    int = (int <<< fract_size) ||| fract
    {int, exp}
  end

  # Gets binary digits count
  defp size(0), do: 0
  defp size(x) do
    round(Float.floor(:math.log2(x))) + 1
  end

  # Implements algorithm from this source https://otvet.mail.ru/question/46720675
  # Перевод из десятичной системы счисления в двоичную и шестнадцатеричную:
  # а) исходная дробь умножается на основание системы счисления, в которую
  #    переводится (2 или 16);
  # б) в полученном произведении целая часть преобразуется в соответствии с
  #    таблицей в цифру нужной системы счисления и отбрасывается – она является
  #    старшей цифрой получаемой дроби;
  # в) оставшаяся дробная часть (это правильная дробь) вновь умножается на
  #    нужное основание системы счисления с последующей обработкой полученного
  #    произведения в соответствии с шагами а) и б);
  # г) процедура умножения продолжается до тех пор, пока ни будет получен
  #    нулевой результат в дробной части произведения или ни будет достигнуто
  #    требуемое количество цифр в результате;
  # д) формируется искомое число: последовательно отброшенные в шаге б) цифры
  #    составляют дробную часть результата, причем в порядке уменьшения старшинства.
  #
  # fb_fract_binary returns:
  # {fractional_result, overall_digits_count}
  defp fb_fract_binary(0.0, _, result), do: result
  defp fb_fract_binary(_, 0, result),   do: result
  defp fb_fract_binary(fract, precision, {result, digits} \\ {0, 0}) do
    fract = fract * 2
    result = result <<< 1
    if fract >= 1 do
      fb_fract_binary(fract - 1, precision - 1, {result + 1, digits + 1})
    else
      fb_fract_binary(fract, precision - 1, {result, digits + 1})
    end
  end
end
