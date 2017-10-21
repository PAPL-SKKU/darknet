#ifndef _LOGGER_HPP_
#define _LOGGER_HPP_

#ifndef __SDSCC__

#include <iostream>
#include <sstream>
#include <string>

/* consider adding boost thread id since we'll want to know whose writting and
 * won't want to repeat it for every single call */

/* consider adding policy class to allow users to redirect logging to specific
 * files via the command line
 */

enum loglevel_e
    {ERROR, WARNING, DEBUG, INFO};

#define __RED__ std::string("\033[1;91m")
#define __END__ std::string("\033[0m")
#define LOG_IF(level, message) (_loglevel > level ? message : "")
class logIt
{
public:
    logIt(loglevel_e _loglevel = ERROR) {
        char buf[10];
        sprintf(buf, "%d", __LINE__);
        _buffer << strmap[_loglevel]
            << LOG_IF(WARNING,  std::string(" [") +
                             std::string(__FILE__) +
                             std::string(", ") +
                             std::string(buf) +
                             std::string("] "))
            << std::string(
                _loglevel > DEBUG 
                ? (_loglevel - DEBUG) * 4 
                : 1
                , ' ');
            // << _loglevel > DEBUG ? __FILE__ << ", " << __LINE__ << " :" 
    }

    template <typename T>
    logIt & operator<<(T const & value)
    {
        _buffer << value;
        return *this;
    }

    ~logIt()
    {
        _buffer << std::endl;
        // This is atomic according to the POSIX standard
        // http://www.gnu.org/s/libc/manual/html_node/Streams-and-Threads.html
        std::cerr << _buffer.str();
    }

private:
    std::ostringstream _buffer;
    static std::string strmap[4];
};

std::string logIt::strmap[4] = {"\033[1;91mERROR!  \033[0m",
                                "\033[1;93mWARNING \033[0m",
                                "\033[1;92mDEBUG   \033[0m",
                                "\033[0;36mINFO    \033[0m"};

extern loglevel_e loglevel;

#define LOG(level) \
if (level > loglevel) ; \
else logIt(level)

#ifdef __DEBUG_ON__
loglevel_e loglevel = DEBUG;
#else
loglevel_e loglevel = ERROR;
#endif

#else // for SDSoC compiler discard all stuff
#define LOG(level) std::cerr
#endif // __SDSCC__

#endif
