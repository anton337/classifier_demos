#ifndef TEXT_H
#define TEXT_H

#include <ostream>

namespace Color {
  enum Code {
    FG_RED      = 31,
    FG_GREEN    = 32,
    FG_BLUE     = 34,
    FG_DEFAULT  = 39,
    BG_RED      = 41,
    BG_GREEN    = 42,
    BG_BLUE     = 44,
    BG_DEFAULT  = 49
  };
  class Modifier {
    Code code;
  public:
    Modifier(Code pCode) : code(pCode) {}
    friend std::ostream&
    operator<<(std::ostream& os, const Modifier& mod) {
      return os << "\033[" << mod.code << "m";
    }
  };
}

namespace General {
  class Clear {
  public:
    Clear() {}
    friend std::ostream&
    operator<<(std::ostream& os, const Clear& mod) {
      return os << "\x1B[2J\x1B[H";
    }
  };
}

/*
 cout << "\033[1;31mbold red text\033[0m\n";

 foreground background
 black        30         40
 red          31         41
 green        32         42
 yellow       33         43
 blue         34         44
 magenta      35         45
 cyan         36         46
 white        37         47

 reset             0  (everything back to normal)
 bold/bright       1  (often a brighter shade of the same colour)
 underline         4
 inverse           7  (swap foreground and background colours)
 bold/bright off  21
 underline off    24
 inverse off      27
*/

//#include <iostream>
//using namespace std;
//int main() {
//  Color::Modifier red(Color::FG_RED);
//  Color::Modifier def(Color::FG_DEFAULT);
//  General::Clear clear;
//  cout << clear;
//  cout << "This ->" << red << "word" << def << "<- is red." << endl;
//}

#endif

