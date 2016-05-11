#pragma once
#include <vector>
#include <string>
#include <set>

std::vector<std::string> split(std::string s, char delimiter = ' ')
{
  std::vector<std::string> result;
  s += delimiter;
  for (int i = 0, last = -1; i < s.size(); ++i)
  {
    if (s[i] == delimiter)
    {
      if (last >= 0)
        result.emplace_back(s, last, i - last);
      last = -1;
    }
    else
    {
      if (last == -1)
        last = i;
    }
  }
  return result;
}

std::vector<std::string> split_words(std::string s, const size_t length = 0)
{
  std::vector<std::string> result;
  s += ' ';
  for (int i = 0, last = -1; i < s.size(); ++i)
  {
    if (((s[i]>>7) == 0 && isalnum(s[i])) || s[i] == '\'')
    {
      if (last == -1)
        last = i;
      continue;
    }
    if (last >= 0)
    {
      result.emplace_back(s, last, i - last);
      if (result.size() == length)
        break;
      last = -1;
    }
    if ((s[i]>>7) == 0 && isspace(s[i]))
    {
      continue;
    }
    result.emplace_back(s, i, 1);
    if (result.size() == length)
      break;
  }
  while (result.size() < length)
    result.push_back("<dummy>");
  return result;
}

// vector of x,y pairs
float integral(std::vector<std::pair<float, float>> coords)
{
  std::sort(coords.begin(), coords.end());
  float result = 0;
  std::pair<float, float> last(0, 0);
  for (const auto& coord : coords)
  {
    result += (coord.first - last.first) * (coord.second + last.second);
    last = coord;
  }
  return result * 0.5;
}