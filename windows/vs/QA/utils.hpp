#pragma once
#include <vector>
#include <string>
#include <set>

std::vector<std::string> split_words(std::string s)
{
  std::vector<std::string> result;
  s += ' ';
  for (int i = 0, last = -1; i < s.size(); ++i)
  {
    if (isalnum(s[i]) || s[i] == '\'')
    {
      if (last == -1)
        last = i;
      continue;
    }
    if (last >= 0)
    {
      result.emplace_back(s, last, i - last);
      last = -1;
    }
    if (isspace(s[i]))
    {
      continue;
    }
    result.emplace_back(s, i, 1);
  }
  return result;
}