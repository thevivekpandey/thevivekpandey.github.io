#include <iostream>
#include <vector>
using namespace std;

int N = 2e2;
vector<vector<int>>pfac(N + 1);

int main() {
   for (int i = 2; i <= N; i++ ) {
      if (pfac[i].empty()) {
         for (int j = i; j <= N; j += i) {
            pfac[j].push_back(i);
         }
      }
   }
   /*for (int i = 2; i <= N; i++) {
      cout << "primes factors of " << i << ": ";
      for (int j = 0; j < pfac[i].size(); j++) {
         cout << pfac[i][j] << ", ";
      }
      cout << "\n";
   }*/
}
