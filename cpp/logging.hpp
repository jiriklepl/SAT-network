#pragma once

#include <iosfwd>
#include <string>

class Logger {
public:
    explicit Logger(int verbosity, std::ostream &out);

    [[nodiscard]] bool enabled(int level) const;
    void log(int level, const std::string &message) const;

private:
    int verbosity_ = 1;
    std::ostream *out_ = nullptr;
};
