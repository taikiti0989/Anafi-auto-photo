#include <kl_evaluation/vbem.h>

void getSortOrderEM(const vector<double>& values, vector<size_t>& order, bool descending)
{
    /* 1. store sorting order in 'order_pair' */
    vector<std::pair<size_t, vector<double>::const_iterator> > order_pair(values.size());

    size_t n = 0;
    for (vector<double>::const_iterator it = values.begin(); it != values.end(); ++it, ++n)
        order_pair[n] = make_pair(n, it);

    std::stable_sort(order_pair.begin(),order_pair.end(),orderingSorterEM());
    if (descending == false) std::reverse(order_pair.begin(),order_pair.end());

    vector<size_t>(order_pair.size()).swap(order);
    for (size_t i = 0; i < order_pair.size(); ++i)
    {
        order[i] = order_pair[i].first;
    }
}
