/*
 * OrderedLink.h
 *
 *  Created on: Jul 25, 2015
 *      Author: vital
 */

#ifndef ORDEREDLINK_H_
#define ORDEREDLINK_H_

#include <set>
#include "LinkCu.h"

namespace PoliFitted {

struct OrderedLinkComp {
  bool operator() (const LinkCu* lhs, const LinkCu* rhs) const
  {return lhs->getSize() > rhs->getSize();}
};

typedef std::multiset<LinkCu*,OrderedLinkComp> OrderedLink;

} /* namespace PoliFitted */
#endif /* ORDEREDLINK_H_ */
