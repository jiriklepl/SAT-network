#include "logging.hpp"

#include <ostream>

Logger::Logger(int verbosity, std::ostream &out) : verbosity_(verbosity), out_(&out) {}

bool Logger::enabled(int level) const {
    return out_ != nullptr && verbosity_ >= level;
}

void Logger::log(int level, const std::string &message) const {
    if (enabled(level)) {
        *out_ << message << "\n";
    }
}
