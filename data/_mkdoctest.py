class B:
    pass

class A:
    """
    title: 1
    ----

    title: 2
    ----

    title: 3
    ----
    * 1
    * 2

    # title: 4
    ## sub
    ### sub
    * 1
    * 2

    """
    def func(self, a:int, b:B):
        """
        # title
        ------

        -----

        Parameters
        -----
        a : int
            12345
        b : `B`
            An instance of :class:`B`.

        Returns
        -----
        int
            12345

        Examples
        --------
        >>> example_function(2, 3)
        5
                
        """
        pass  