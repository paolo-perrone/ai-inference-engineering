# What a correct patch does

The agent's patch is correct when all four hold. This is the scoring rule, so a run can be
called complete or not without a human reading the diff.

1. **`fetch_user` retries transient failures and does not retry a 4xx.** A 404 is an answer,
   not a fault; retrying it burns the upstream and returns the same thing.
2. **Backoff is bounded and it waits between attempts.** A tight loop is not a retry, it is
   the same failure three times faster.
3. **The final failure still raises.** Swallowing it returns `None` into calling code that
   expects a record, which turns one visible failure into a later invisible one.
4. **`fetch_users` is unchanged.** It composes `fetch_user`, so the fix belongs one level
   down. A patch that adds retry in both places is doing the work twice.

A run that satisfies all four is one completed task. That count is the denominator of every
cost number in the book.
